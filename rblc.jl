import Base.range
struct RangeHolder{T}
  minimum::T
  maximum::T
  RangeHolder{T}(min,max) where {T<:Any} = min >= max ? error("Minimum must be smaller than maximum.") : new(min,max)
end

function range(rg::RangeHolder{Integer})
  return rg.maximum-rg.minimum
end

function range(rg::RangeHolder{Date})
  return Dates.value(rg.maximum-rg.minimum)
end

function readDF(df)
  rows=size(df,1)
  inpu=Array{Date}(rows)
  for i in 1:rows
    inpu[i]=Date(df[:datey][i],df[:datem][i],df[:dated][i])
  end
  return inpu, df[:subcount]
end

normSub(sub::Integer,rgh::RangeHolder{Integer})=sub/range(rgh)
invSub(nrms::Real,rgh::RangeHolder{Integer})=floor(nrms*range(rgh))

"""
    real2normDate(date,dr)

Compute the normalized date of date respect to dr.
# Arguments
- `date::Date`: date to be normalized.
- `dr::RangeHolder{Date}`: RangeHolder of the start and end dates.
"""
function real2normDate(date::Date,dr::RangeHolder{Date})
  intDate=Dates.date2epochdays(date)-Dates.date2epochdays(dr.minimum)
  ret=intDate/range(dr)
  return ret
end

"""
    norm2realDate(normedDate,dr)

Compute real date of normedDate respect to minDate and drange.
# Arguments:
- `normedDate::Real`: normalized date.
- `dr::RangeHolder{Date}`: RangeHolder of the start and end dates.

"""
function norm2realDate(normedDate::Real,dr::RangeHolder{Date})
  intdata=floor(normedDate*range(dr))
  intDate=intdata+Dates.date2epochdays(dr.minimum)
  return Dates.epochdays2date(intDate)
end

"""
    createNNinput(df,sr::RangeHolder{Integer},dr::RangeHolder{Date})

Create the input-output array of the NN  with df respect to subrange and
mindate and drange.
# Arguments
- `df::DataFrames.DataFrame`: input dataframe.
- `sr::RangeHolder{Integer}`: subscribers range.
- `dr::RangeHolder{Date}`: Date range.
"""
function createNNinput(df,sr::RangeHolder{Integer},dr::RangeHolder{Date})
#Inputs: readed dataframe, sub range, minimum date, date range
  rows=size(df,1)
  inputArray=zeros(Float64,rows)
  outputArray=zeros(Float64,rows)
  for i in 1:rows
    outputArray[i]=normSub(df[:subcount][i],sr)
    currentDate=Date(df[:datey][i],df[:datem][i],df[:dated][i])
    inputArray[i]=real2normDate(currentDate,dr)
  end
  return inputArray, outputArray
end

#Knet functions
predict_lin(W,x)=W[1]*x+W[2]
loss_lin(W,x,y)=sum(abs2,y-predict_lin(W,x))/ size(y,2)
lossgrad_lin=grad(loss_lin)

"""
    learnLinPars(iters,daf,lr,sranges,mindates,dranges;printNum=100)

Calculating the fitting parameters if linear function.
# Arguments
- `iters:Integer`: number of iterations.
- `daf::DataFrames.DataFrame`: input dataframe.
- `lr:Float64`: learning rate.
- `SR::RangeHolder{Integer}`: subscribers range.
- `DR::RangeHolder{Date}`: Date range.
- `printNum::Integer`: how often to print the loss, default is 100
"""
function learnLinPars(iters,daf,lr,SR::RangeHolder{Integer},DR::RangeHolder{Date};printNum=100)
  W=rand(2)
  inp,outp=createNNinput(daf,SR,DR)
  start_loss=loss_lin(W,inp[1],outp[1])
  println("Start loss is $start_loss.")
  setSize=size(inp,1)
  wSize=size(W,1)
  for k in 1:iters
    for j in 1:setSize
      dw=lossgrad_lin(W,inp[j],outp[j])
      for i in 1:wSize
        W[i]-=lr*dw[i]
      end
    end
    if k%printNum==0
      kthLoss=0
      for i in 1:setSize
        kthLoss+=loss_lin(W,inp[i],outp[i])
      end
      avgLoss=mean(kthLoss)
      println("Average loss in $k-th iteration is: $avgLoss")
    end
  end
  return W
end

function subFromDate(w,d::Date,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  fld=real2normDate(d,dr)
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_lin(w,fld),sr)
end

function subFromDate(w,fld::Real,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_lin(w,fld),sr)
end

function resLinNN(w,sr::RangeHolder{Integer},dr::RangeHolder{Date},df)
  oriDates,oriSubs=readDF(df)
  flDates=collect(linspace(0,1,50))
  pDates=[norm2realDate(x,dr) for (i,x) in enumerate(flDates)]
  pSubs=[subFromDate(w,x,sr,dr) for (i,x) in enumerate(flDates)]
  return pDates,pSubs,oriDates,oriSubs
end

function whenGivenSub(sub::Integer,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  if sub<s.minimum || sub>s.maximum
    error("Given sub is out of range!")
  end
  nS=normSub(sub,s)
  nD=(nS-W[2])/W[1]
  rD=norm2realDate(nD,d)
  println("Given $sub subscribers will be achieved on $rD.")
  return rD
end

function plotLinNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df)
  pD,pS,oD,oS=resLinNN(w,s,d,df);
  xt=d.minimum:Dates.Day(200):d.maximum
  yt=s.minimum:5000:s.maximum
  plot(pD,pS,legend=:bottomright,xticks=xt,yticks=yt,xlabel="Date",ylabel="Subscribers count",label="Predicted values",size=(800,600),dpi=200,title="Linear approximation")
  scatter!(oD,oS,label="Teaching set")
end

function plotLinNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS)
  gD=whenGivenSub(gS,w,s,d)
  plotLinNN(w,s,d,df)
  scatter!([gD,gD],[gS,gS], label="Given subs")
end
