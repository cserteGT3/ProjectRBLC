numchop{T <: Real}(x::T, δ = eps(T)) = abs(x) > δ ? x : zero(T)

function numchop{T <: Real}(x::Complex{T}, δ = eps(T))
  ichop = numchop(imag(x), δ)
  rchop = numchop(real(x), δ)
  if ichop == zero(T)
    rchop
  else
    complex(rchop, ichop)
  end
end

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

#Linear functions
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
function learnLinPars(iters,daf,lr,SR::RangeHolder{Integer},DR::RangeHolder{Date};printNum=500)
  W=rand(2)
  inp,outp=createNNinput(daf,SR,DR)
  start_loss=loss_lin(W,inp[1],outp[1])
  println("Linear start loss is $start_loss.")
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
      println("Average loss with linear approximation in $k-th iteration is: $avgLoss")
    end
  end
  return W
end

function linSubFromDate(w,d::Date,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  fld=real2normDate(d,dr)
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_lin(w,fld),sr)
end

function linSubFromDate(w,fld::Real,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_lin(w,fld),sr)
end

function resultLinNN(w,sr::RangeHolder{Integer},dr::RangeHolder{Date},df)
  oriDates,oriSubs=readDF(df)
  flDates=collect(linspace(0,1,50))
  pDates=[norm2realDate(x,dr) for (i,x) in enumerate(flDates)]
  pSubs=[linSubFromDate(w,x,sr,dr) for (i,x) in enumerate(flDates)]
  return pDates,pSubs,oriDates,oriSubs
end

function linWhenGivenSub(sub::Integer,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  if sub<s.minimum || sub>s.maximum
    error("Given sub is out of range!")
  end
  nS=normSub(sub,s)
  nD=(nS-w[2])/w[1]
  rD=norm2realDate(nD,d)
  println("Given $sub subscribers based on linear approximation will be achieved on $rD.")
  return rD
end

function linWhenGivenSub(sub::Array,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  return [linWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(sub)]
end

function plotLinNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df)
  pD,pS,oD,oS=resultLinNN(w,s,d,df);
  xt=d.minimum:Dates.Day(200):d.maximum
  yt=s.minimum:5000:s.maximum
  plot(pD,pS,legend=:bottomright,xticks=xt,yticks=yt,xlabel="Date",ylabel="Subscribers count",label="Predicted values",size=(800,600),dpi=200,title="Linear approximation")
  scatter!(oD,oS,label="Teaching set")
end

function plotLinNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS)
  gD=linWhenGivenSub(gS,w,s,d)
  plotLinNN(w,s,d,df)
  scatter!([gD,gD],[gS,gS], label="Given subs")
end

function plotLinNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS::Array)
  gD=[linWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(gS)]
  plotLinNN(w,s,d,df)
  scatter!(gD,gS, label="Given subs")
end

#Exponential functions
predict_exp(W,x)=W[1]*exp(x)+W[2]
loss_exp(W,x,y)=sum(abs2,y-predict_exp(W,x))/ size(y,2)
lossgrad_exp=grad(loss_exp)

function learnExpPars(iters,daf,lr,SR::RangeHolder{Integer},DR::RangeHolder{Date};printNum=100)
  W=rand(2)
  inp,outp=createNNinput(daf,SR,DR)
  start_loss=loss_exp(W,inp[1],outp[1])
  println("Exponential start loss is $start_loss.")
  setSize=size(inp,1)
  wSize=size(W,1)
  for k in 1:iters
    for j in 1:setSize
      dw=lossgrad_exp(W,inp[j],outp[j])
      for i in 1:wSize
        W[i]-=lr*dw[i]
      end
    end
    if k%printNum==0
      kthLoss=0
      for i in 1:setSize
        kthLoss+=loss_exp(W,inp[i],outp[i])
      end
      avgLoss=mean(kthLoss)
      println("Average loss with linear exponential in $k-th iteration is: $avgLoss")
    end
  end
  return W
end

function expSubFromDate(w,d::Date,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  fld=real2normDate(d,dr)
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_exp(w,fld),sr)
end

function expSubFromDate(w,fld::Real,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_exp(w,fld),sr)
end

function resultExpNN(w,sr::RangeHolder{Integer},dr::RangeHolder{Date},df)
  oriDates,oriSubs=readDF(df)
  flDates=collect(linspace(0,1,50))
  pDates=[norm2realDate(x,dr) for (i,x) in enumerate(flDates)]
  pSubs=[expSubFromDate(w,x,sr,dr) for (i,x) in enumerate(flDates)]
  return pDates,pSubs,oriDates,oriSubs
end

function expWhenGivenSub(sub::Integer,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  if sub<s.minimum || sub>s.maximum
    error("Given sub is out of range!")
  end
  nS=normSub(sub,s)
  preV=(nS-w[2])/w[1]
  if preV<0
    error("Argument of logarithmus naturalis is negative: $preV")
  end
  nD=log(preV)
  rD=norm2realDate(nD,d)
  println("Given $sub subscribers based on exponential approximation will be achieved on $rD.")
  return rD
end

function expWhenGivenSub(sub::Array,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  return [expWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(sub)]
end

function plotExpNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df)
  pD,pS,oD,oS=resultExpNN(w,s,d,df);
  xt=d.minimum:Dates.Day(200):d.maximum
  yt=s.minimum:5000:s.maximum
  plot(pD,pS,legend=:bottomright,xticks=xt,yticks=yt,xlabel="Date",ylabel="Subscribers count",label="Predicted values",size=(800,600),dpi=200,title="Exponential approximation")
  scatter!(oD,oS,label="Teaching set")
end

function plotExpNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS)
  gD=expWhenGivenSub(gS,w,s,d)
  plotExpNN(w,s,d,df)
  scatter!([gD,gD],[gS,gS], label="Given subs")
end

function plotExpNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS::Array)
  gD=[expWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(gS)]
  plotExpNN(w,s,d,df)
  scatter!(gD,gS, label="Given subs")
end

#Quadratic functions
predict_quad(W,x)=W[1]*x^2+W[2]*x+W[3]
loss_quad(W,x,y)=sum(abs2,y-predict_quad(W,x))/ size(y,2)
lossgrad_quad=grad(loss_quad)

function learnQuadPars(iters,daf,lr,SR::RangeHolder{Integer},DR::RangeHolder{Date};printNum=100)
  W=rand(3)
  inp,outp=createNNinput(daf,SR,DR)
  start_loss=loss_quad(W,inp[1],outp[1])
  println("Quadratic start loss is $start_loss.")
  setSize=size(inp,1)
  wSize=size(W,1)
  for k in 1:iters
    for j in 1:setSize
      dw=lossgrad_quad(W,inp[j],outp[j])
      for i in 1:wSize
        W[i]-=lr*dw[i]
      end
    end
    if k%printNum==0
      kthLoss=0
      for i in 1:setSize
        kthLoss+=loss_quad(W,inp[i],outp[i])
      end
      avgLoss=mean(kthLoss)
      println("Average loss with quadratic approximation in $k-th iteration is: $avgLoss")
    end
  end
  return W
end

function quadSubFromDate(w,d::Date,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  fld=real2normDate(d,dr)
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_quad(w,fld),sr)
end

function quadSubFromDate(w,fld::Real,sr::RangeHolder{Integer},dr::RangeHolder{Date})
  if fld<0 || fld>1
    error("Float date is out of range: $fld")
  end
  return invSub(predict_quad(w,fld),sr)
end

function resultQuadNN(w,sr::RangeHolder{Integer},dr::RangeHolder{Date},df)
  oriDates,oriSubs=readDF(df)
  flDates=collect(linspace(0,1,50))
  pDates=[norm2realDate(x,dr) for (i,x) in enumerate(flDates)]
  pSubs=[quadSubFromDate(w,x,sr,dr) for (i,x) in enumerate(flDates)]
  return pDates,pSubs,oriDates,oriSubs
end

function quadWhenGivenSub(sub::Integer,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  if sub<s.minimum || sub>s.maximum
    error("Given sub is out of range!")
  end
  nS=normSub(sub,s)
  rts=roots([w[3]-nS,w[2],w[1]])
  realRts=map(numchop,rts)
  boolVal=falses(size(realRts,1))
  for (ind,val) in enumerate(realRts)
    if val<=1 && val>=0
      boolVal[ind]=true
    end
  end
  sumOfbV=sum(boolVal)
  if sumOfbV==0
    error("Didn't find solution for quadratic equation: $realRts")
  elseif sumOfbV==1
    for (ind,val) in enumerate(boolVal)
      if val
        rd=norm2realDate(realRts[ind],d)
        println("Given $sub subscribers based on quadratic approximation will be achieved on $rd.")
        return rd
      end
    end
  else
    error("Multiple valid solutions found for quadratic equation: $realRts")
  end
end

function quadWhenGivenSub(sub::Array,w,s::RangeHolder{Integer},d::RangeHolder{Date})
  return [quadWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(sub)]
end

function plotQuadNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df)
  pD,pS,oD,oS=resultQuadNN(w,s,d,df);
  xt=d.minimum:Dates.Day(200):d.maximum
  yt=s.minimum:5000:s.maximum
  plot(pD,pS,legend=:bottomright,xticks=xt,yticks=yt,xlabel="Date",ylabel="Subscribers count",label="Predicted values",size=(800,600),dpi=200,title="Quadratic approximation")
  scatter!(oD,oS,label="Teaching set")
end

function plotQuadNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS::Real)
  gD=quadWhenGivenSub(gS,w,s,d)
  plotQuadNN(w,s,d,df)
  scatter!([gD,gD],[gS,gS], label="Given subs")
end

function plotQuadNN(w,s::RangeHolder{Integer},d::RangeHolder{Date},df,gS::Array)
  gD=[quadWhenGivenSub(val,w,s,d) for (ind,val) in enumerate(gS)]
  plotQuadNN(w,s,d,df)
  scatter!(gD,gS, label="Given subs")
end

function learnNNPars(tip,iters,daf,lr,SR::RangeHolder{Integer},DR::RangeHolder{Date};printNum=500)
  if tip==:lin || tip==:exp
    W=rand(2)
  elseif tip==::quad
    W=rand(3)
  end

  inp,outp=createNNinput(daf,SR,DR)
  start_loss=loss_lin(W,inp[1],outp[1])
  println("Linear start loss is $start_loss.")
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
      println("Average loss with linear approximation in $k-th iteration is: $avgLoss")
    end
  end
  return W
end
