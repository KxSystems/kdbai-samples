dst:`:demo_hdb    / database root

numpart:10;
ed:2024.08.31; / end date the last date partition
dts:{[ed;n] reverse n#d where 1<mod[d:ed-til 2*n]7}[ed;numpart];
bgn:first dts;
end:last dts;
bgntm: 09:30:00.0;  / exchange open time
endtm: 16:00:00.0;  / exchange close time

nt:100000;            / trades per stock per day
qpt:1;

\S 104831           / random seed

/ utils
pi:acos -1
accum:{prds 1.0,-1 _ x}
int01:{(til x)%x-1}
limit:{(neg x)|x & y}
minmax:{(min x;max x)}
normalrand:{(cos 2 * pi * x ? 1f) * sqrt neg 2 * log x ? 1f}
rnd:{0.01*floor 0.5+x*100}
xrnd:{exp x * limit[2] normalrand y}
randomize:{value "\\S ",string "i"$0.8*.z.p%1000000000}
shiv:{(last x)&(first x)|asc x+-2+(count x)?5}
vol:{10+`int$x?90}
vol2:{x?100*1 1 1 1 2 2 3 4 5 8 10 15 20}

/ =========================================================
choleski:{
 n:count A:x+0.0;
 if[1>=n;:sqrt A];
 p:ceiling n%2;
 X:p#'p#A;
 Y:p _'p#A;
 Z:p _'p _A;
 T:(flip Y) mmu inv X;
 L0:n #' (choleski X) ,\: (n-1)#0.0;
 L1:choleski Z-T mmu Y;
 L0,(T mmu p#'L0),'L1}

/ =========================================================
/ paired correlation, matrix of variates, min 0.1 coeff
choleskicor:{
 x:"f"$x;y:"f"$y;
 n:count y;
 c:0.1|(n,n)#1.0,x,((n-2)#0.0),x;
 (choleski c) mmu y}

/ =========================================================
/ volume profile - random times, weighted toward ends
/ x=count
volprof:{
 p:1.75;
 c:floor x%3;
 b:(c?1.0) xexp p;
 e:2-(c?1.0) xexp p;
 m:(x-2*c)?1.0;
 {(neg count x)?x} m,0.5*b,e}

/ =========================================================
write:{
 t:.Q.en[dst] update sym:`p#sym from `sym xasc y;
  (` sv dst,`$x) set t}

/ symbol data for tick demo
sn:2 cut (
  `AAPL;218;
  `TSLA;210;
  `GOOG;135;
  `AMZN;145;
  `MSFT;332;
  `NVDA;458;
  `META;517;
  `NFLX;412;
  `ADBE;555;
  `PYPL;65
  )

s:first each sn
p:last each sn

/ gen
vex:1.0005         / average volume growth per day
ccf:0.5            / correlation coefficient

/ =========================================================
/ qx index, qb/qbb/qa/qba margins, qp price, qn position
batch:{[x;len]
  p0:prices[;x];
  p1:prices[;x+1];
  d:xrnd[0.0003] len;
  qx::0N?raze {(floor len%cnt)#x} each til cnt;
  qb::rnd len?1.0;
  qa::rnd len?1.0;
  qbb::qb & -0.02 + rnd len?1.0;
  qba::qa & -0.02 + rnd len?1.0;
  n:where each qx=/:til cnt;
  s:p0*accum each d n;
  s:s + (p1-last each s)*{int01 count x} each s;
  qp::len#0.0;
  (qp n):rnd s;
  qn::0
 }

/ =========================================================
/ constrained random walk
/ x max movement per step
/ y max movement at any time (above/below)
/ z number of steps
cgen:{
  m:reciprocal y;
  while[any (m>p) or y<p:prds 1.0+x*normalrand z];
  p}

/ =========================================================
getdates:{
 b:x 0;
 e:x 1;
 d:b + til 1 + e-b;
 d where 5> d-`week$d
 }

/ =========================================================
makeprices:{
 r:cgen[0.0375;3] each cnt#nd;
 r:choleskicor[ccf;1,'r];
 (p % first each r) * r *\: 1.1 xexp int01 nd+1}

/ =========================================================
/ day volumes
makevolumes:{x#1}

/ main
cnt:count s
dates:getdates bgn,end
nd:count dates

prices:makeprices nd + 1
volumes:floor (cnt*nt) * makevolumes nd

calcvols:{[vols;x]
  len::vols x;
  batch[x;len];
 }

day:{

  sa:string dx:dates x;
  show"Creating Partition for ",sa;
  / trade table
  calcvols[volumes;x];
  maxtime:max floor (endtm-bgntm)*volprof count qx;
  tvol:$[(count qx)>maxtime;(count qx);(neg count qx)];
  r:asc bgntm+tvol?maxtime;
  cx:0N?raze {(floor len%qpt)#x} each til qpt;
  cn:count n:where cx<qpt;
  t:([]sym:s qx n;time:dx+shiv r n;price:qp n;size:vol cn);
  write[sa,"/trade/";t]
 }

day each til nd;

1"\n\nCreated Sample HDB.\n\n";