#done
function unpackDates(dateArray)
  par=zeros(Float64,length(dateArray))
  val=zeros(Float64,length(dateArray))
  for (i,x) in enumerate(dateArray)
    par[i]=dateToNN(x[1])
    val[i]=subsToNN(x[2])
  end
  return par,val
end

#done
function nnDateToRealDate(nndate)
  floatdate=nndate*dateLength
  intdate=convert(Int64,trunc(floatdate))+Dates.value(maxDate)
  dtt=Date(Dates.UTD(intdate))
  return dtt
end

function gen_subsToNNDate(w,subs)
  nnSub=subsToNN(subs)
  x=(nnSub-w[2])/w[1]
  return x,nnSub
end

function realSubsToRealDate(pars,subs)
  nndate,nnSuss=gen_subsToNNDate(pars,subs)
  ret=nnDateToRealDate(nndate)
  return ret
end

function initVals()
  rblc_data=[(Date(2016,12,24),2^7),(Date(2017,1,7),2^8),
  (Date(2017,3,10),2^9),(Date(2017,5,30),2^10),
  (Date(2017,6,20),2^11),(Date(2017,9,10),2^12)]
  inn,outt=unpackDates(rblc_data)
  return inn, outt
end

predict_lin(W,x)=W[1]*x+W[2]
loss_lin(W,x,y)=sum(abs2,y-predict_lin(W,x))/ size(y,2)
lossgrad_lin=grad(loss_lin)

function calc_linparam(iteratornum,lr,printNum)
  wInit=rand(2)
  inData,outData=initVals()
  start_loss=loss_lin(wInit,inData[1],outData[1])
  dataSize=length(inData)
  println("Initial loss: $start_loss")
  for k in 1:iteratornum
    for j in 1:dataSize
     dw=lossgrad_lin(wInit,inData[j],outData[j])
      for i in 1:length(wInit)
        wInit[i]-=lr*dw[i]
      end
    end
    if k%printNum==0
      kthloss=loss_lin(wInit,inData[1],outData[1])
      println("loss after $k iteration: $kthloss")
    end
  end
  return inData,outData,wInit
end

function gen_NNPlotData(w,array_size)
  xs=collect(linspace(0,1,array_size))
  ys=zeros(Float64,array_size)
  for (i,val) in enumerate(xs)
    ys[i]=predict_lin(w,val)
  end
  return xs,ys
end

#calc_linparam(10,.1,5)
#inDots,outDots,Wk=calc_linparam(1000,.1,200)
#xs,ys=gen_NNPlotData(Wk,50)
#subX,subY=gen_subsToNNDate(Wk,2^13)
#plot(xs,ys)
#plot!(inDots,outDots,line=:scatter)
#plot!(subX,subY,line=:scatter,c=:green)
#savefig("valami.png")

If I come here and write some things.
