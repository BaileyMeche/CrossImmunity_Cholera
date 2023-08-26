function []=Cholera_epitopeStochTrial_baileyedits


% Parameters:

% B: host-host transmission
% D: environment-host transmission
% V: rate of host recovery
% M: host population turnover rate
% G: degree of cross-immunity (ranges between 0 and 1)
% A: host shedding rate of pathogen
% r: pathogen decay rate
% D0: degree of seasonality (ranges between 0 and 1)

%%%%%TRANSLATION from LeahSrotypemodel to Cholera_epitopeStochTrial
% D = bw(1+amp*c)   = environment-host transmission
% M = mu            = host population turnover rate
% B = bi            = host-host transmission
% V = gamma         = rate of host recovery
% G = chi           = degree of cross-immunity (ranges between 0 and 1)         
% A = xi            = host shedding rate of pathogen
% r = nu 

%Parameters from Koelle: V, M, G, B; D0 from Sasaki_2002; A, D, r from
%Bani: A multiplied by 10^5 since (normalized, time adjusted to days) B from Koelle is on order of
%10^-1 and Bani B is on order of 10^-7 without normalization; we use the D
%from Bani which isn't affected by normalization

%Parameters working well on 3/15: responding well to tail ends of data
%% mu=  9*10^(-5); 
chi=0.9;



%%%% Parameterization adjustment from LeahSerotypemodel

%bi=2.9*((1/9.5)+(1/(38.8*365.2422))); 
bi = 2.5*(10)^(-6)  ;

%D=1.07*10^(-5);
mu=  9*10^(-5);  
%mu = 1/(38.8*365.2422);
%mu=1/(365*61.2); % adjustment


%V = 1/9.5; A = 10^6; r=10; D0=0.1;

%xi = 10

xi=10^6;              %%%%ADJUSTED 

amp=0.15;
%amp = 10^(-3);
amp2=0.5;
amp3=0.5;
%
weight=1e-5;
gamma=1/9;
%muv(end)=10*mu;

%%%%%%% continue original model

alpha0=1/(365*1);
chi0=0.9;       %chi0=0.9;

%xir=1;
K=1;
N=9923.24e3;
sigma=0.36;
zeta=1;

 sigs=.98;
 r=2;
%r=2;
% rP=r*xi
nu=1/30;
R0h=0.71*4;  %%%%
R0c=0.84*4;
bi0=(R0h)*(gamma+mu)
bw0=(R0c/xi)*(nu*(gamma+mu))
R0h=(bi0)/(gamma+mu)
R0c=(bw0*xi)/(nu*(gamma+mu))
nuP=nu*xi
bi0=(R0h)*(gamma+mu)
bw0=(R0c/xi)*(nu*(gamma+mu))
R0g=r/nu
R0=1/2*(R0h+R0g+sqrt((R0h-R0g)^2+4*R0c))

extinctlim=1;
  
X=zeros(7,1);
X(1)=1;
%X=[b/a, 26, 0, 0, 0,0,0 0, 0,.1,.1,0.1]';

%X(2^(d)+2)=50;
  
%X(m+2:m+d+1)=0.1*ones(d,1);
  
  
  
t_r=1;
%pop_c=0.0001;
  
L2=length(X);
  
  
%numreac1=2^numEpi;
  
%filename = 'monthly_cholera_case_reports_2017_check.xlsx';
filename = 'monthly_cholera_case_reports_2017.xlsx';
xlRange = 'A2:B88';
dataArray = xlsread(filename,xlRange);
%I=1:1:86;
Tc=dataArray(:,1)
 Cases = dataArray(:,2);
 Tc=Tc-40482+30;
 Tc(1)

 
 I10g=1/N*Cases(1)*.15;
 X(2)=I10g;
X(3)=.0001*X(2);
X(6)=xi/nu*X(2);
X(7)=xi/nu*X(3);

%v1=A+eye(numreac1);
filename = 'haiti_rainfall_water_chl_cases_data.xlsx'; 
%filename = '/Users/cxb0559/Dropbox/ImmuneAgeModel/CholeraData/haiti_rainfall_water_chl_cases_data.xlsx';

 xlRange = 'A21:G145';
dataArray_v = xlsread(filename,xlRange);
%dataArray_v = readmatrix(filename);
I=1:2:145-20;
%Tr=dataArray_v(I,1);
Tr = dataArray_v(I,1)

%%% THESE RAINFALL CASES ARE NEVER USED.. 
Cases_v=dataArray_v(I,5)
Cases_v(isnan(Cases_v))=0;

Precip=dataArray_v(I,2)

Tr=Tr+365*(1903+11/12-2010.75);
Try=mod(Tr-87,365);
 Trm=ceil(Try/30.4);
 Tr(1)
 Prm=zeros(12,1);
 
 
 for ik=1:12
     Prm(ik)=mean(Precip(Trm==ik));
 end
tsp=[Tr;Tr(end)+(30:30:30*12*10)'];
      Prmm=repmat(Prm,10,1); 
       Ps=[Precip;Prmm];
 
  
 a1 =       116.9 ;
       b1 =      0.6061 ;
       c1 =      0.1992 ;
       a2 =       47.68  ;
       b2 =       6.072  ;
       c2 =       2.198  ;
       a3 =       42.07 ;
       b3 =        1.16  ;
       c3 =       1.719  ;
       a4 =       19.89  ;
       b4 =       7.097 ;
       c4 =       1.161 ;
        
  
 per=3.75;
 cavg=75;
 cmax=164;
  
filename = 'all_vc_sero_info.xlsx';%'/Users/cxb0559/Dropbox/ImmuneAgeModel/CholeraData/SerotypeModel/all_vc_sero_info.xlsx';
xlRange = 'B2:C263';
[~, txt] = xlsread(filename,xlRange);
Serotype = txt(:,1);
Time = str2double(txt(:,2));
Time=365*(Time-2010.75)+30;
% strT = strings([262,1]);

TimeI=zeros(262,1);
TimeO=zeros(262,1);
 for ii=1:262
     if strcmp(Serotype(ii),'Inaba')==1
         TimeI(ii)=Time(ii);
     else TimeO(ii)=Time(ii);
     end
%     Ti=txt(i,1);
%     L=strlength(Ti);
%     strT(i,1)=extractBetween(Ti,L-7,L);
 end

 TimeI(TimeI==0)=[];
 TimeO(TimeO==0)=[];
 
 [NO,edges] = histcounts(TimeO,'BinLimits',[0,max(Time)],'BinWidth',30);
 [NI,edges] = histcounts(TimeI,'BinLimits',[0,max(Time)],'BinWidth',30);
 
 nti=NI+NO;
 pOd=NO./nti;
 
 Mdm=length(diff(Tc));
 
 Nd=length(NO);
 pOi=0.5*ones(Nd,1);
 Md=max(Mdm,Nd);
  CaseMonO=zeros(Md-1,1);
  CaseMonI=zeros(Md-1,1);
  CaseTot=zeros(Md-1,1);
  
%epsilon=.03;
  
  
 %X(2:m+1)=Xv;

%  size(Xv)
%  size(X(m+3:end))
%  size(aE)
%  size(aH.*((bw)*X(m+3:2*m+2)+(bi)*X(2:m+1)))
%  size(xi.*(X(2:m+1)+X(m+3:end).*(aE.*(ones(m,1)-X(m+3:end)/K)-ones(m,1))))
  
%%
  
options=odeset('AbsTol',1e-25*ones(1,7));
  


%DELETED OLD ODE MODEL: BEGIN SYNTHESIZED MODEL

function [dxdt] = myoden(t,x,bi,bw,chi)
   xt=mod(t/365,per);
    c =  ((a1*sin(b1*xt+c1) + a2*sin(b2*xt+c2) + a3*sin(b3*xt+c3) + a4*sin(b4*xt+c4))-cavg)/cmax;

    %TRANSLATION from LeahSrotypemodel to Cholera_epitopeStochTrial
% D = bw(1+amp*c)
% M = mu
% bi = B  
% V = gamma
% chi maybe = G or (1-G)
% A = xi 
% r = nu 


%dxdt(1) = mu - mu*x(1)+alpha*(x(4)+x(5)) - (bw*(1+amp*c)*(x(6)+x(7))+bi*(x(2)+x(3)))*x(1);

%old: dxdt(1) = mu - mu*x(1)+alpha*(x(4)+x(5)) - (bw*(1+amp*c)*(x(6)+x(7))+bi*(x(2)+x(3)))*x(1);
%Leah: %dxdt(1) = -bw*(1+amp*c)*(x(6)+x(7))*x(1) -bi*x(1)*(x(2)+x(3))  +mu(1-x(1));

%works: 
dxdt(1) = -bw*(1+amp*c)*(x(6)+x(7))*x(1) -bi*x(1)*(x(2)+x(3)) +mu*(1-x(1));

%old: dxdt(2) = x(2)*(bi*(x(1)+chi*x(5)) - gamma - mu) +x(6)*bw*(1+amp*c)*(x(1)+chi*x(5))
dxdt(2) = bi*x(1)*x(2) - x(2)*(gamma + mu) + bi*x(2)*x(5)*chi  + bw*(1+amp*c)*chi*x(5)*x(6) + bw*(1+amp*c)*x(1)*x(6);
%dxdt(2) = bi*x(1)*x(2) - x(2)*(gamma + mu) + bi*x(2)*x(5)*(1-chi)  + bw*(1+amp*c)*(1-chi)*x(5)*x(6) + bw*(1+amp*c)*x(1)*x(6);

%old: dxdt(3) = x(3)*(bi*(x(1)+chi*x(4)) - gamma - mu) +x(7)*bw*(1+amp*c)*(x(1)+chi*x(4));
dxdt(3) = x(3)*bi*x(1) + x(3)*bi*chi*x(4) - x(3)*(gamma+mu) + x(7)*bw*(1+amp*c)*x(1) +x(7)*bw*(1+amp*c)*chi*x(4);
%dxdt(3) = x(3)*bi*x(1) + x(3)*bi*(1-chi)*x(4) - x(3)*(gamma+mu) + x(7)*bw*(1+amp*c)*x(1) +x(7)*bw*(1+amp*c)*(1-chi)*x(4);

%dxdt(3) = x(3)*(bi*(x(1)+chi*x(4)) - gamma - mu) +x(7)*bw*(1+amp*c)*(x(1)+chi*x(4));

%old: dxdt(4) = (gamma*x(2)-(mu+alpha+chi*(bi*x(3)+bw*(1+amp*c)*x(7)))*x(4)); %remove alpha*x(4)
%works: dxdt(4) = gamma*x(2) - x(4)*mu - chi*x(4)*(bi*x(3)+bw*(1+amp*c)*x(7));
dxdt(4) = (gamma)*x(2) - chi*bi*x(3)*x(4) - mu*x(4) - bw*(1+amp*c)*chi*x(4)*x(7);
%dxdt(4) = (gamma)*x(2) - (1-chi)*bi*x(3)*x(4) - mu*x(4) - bw*(1+amp*c)*(1-chi)*x(4)*x(7); 

%old: dxdt(5) = (gamma*x(3)-(mu+alpha+chi*(bi*x(2)+bw*(1+amp*c)*x(6)))*x(5));
%works: dxdt(5) = gamma*x(3) - x(5)*mu - chi*x(5)*(bi*x(2)+bw*(1+amp*c)*x(6));
dxdt(5) = (gamma)*x(3) - chi*bi*x(2)*x(5) - mu*x(5) -bw*(1+amp*c)*chi*x(5)*x(6);
%dxdt(5) = (gamma)*x(3) - (1-chi)*bi*x(2)*x(5) - mu*x(5) - bw*(1+amp*c)*(1-chi)*x(5)*x(6);

dxdt(6) = xi*x(2)-nu*x(6);
dxdt(7) = xi*x(3)-nu*x(7);
dxdt=dxdt';
end  

%%%%END SYNTHESIZED MODEL

%% 
%options=odeset('AbsTol', [1e-23,1e-23,1e-23,1e-25,1e-25,1e-25,1e-25]);
t0=0;
tcon=600
tend=max(Tc(end),Time(end));
%tend=1882;
Tspan=t0:1:t0+tend;
    function LL=negloglike(param)
        
        bi=param(1);
         bw=param(2);
          chi=param(3);
           %alpha=param(4);
           I10=param(5);
           X(2)=I10;
           X(3)=.0001*X(2);
X(6)=xi/nu*X(2);
X(7)=xi/nu*X(3);
        
[T1, Z1] = ode45(@(t,x)myoden(t,x,bi,bw,chi),Tspan(1:tcon),X,options); 
    bi=0.825*bi;
    bw=0.825*bw;
    
    
[T2, Z2] = ode45(@(t,x)myoden(t,x,bi,bw,chi),Tspan(tcon:end),Z1(end,:),options); 
    Z=[Z1;Z2(2:end,:)];
    for i=1:Md-1
        CaseMonO(i)=sum(gamma/30*Z((i-1)*30+1:i*30,2))/4*N;
        CaseMonI(i)=sum(gamma/30*Z((i-1)*30+1:i*30,3))/4*N;
       
        CaseTot(i)=(CaseMonO(i)+CaseMonI(i));
        if i<=Nd
            pOi(i)=CaseMonO(i)/(CaseTot(i));
            
        end
      
    end

         %LLh=-NO*log(pOi)-(nti-NO)*log(1-pOi)-sum(log(factorial(nti)))+sum(log(factorial(nti-NO)))+sum(log(factorial(NO)))
% 
%          
%         LLc=-Cases'*log(CaseTot(1:end-1))+sum(CaseTot(1:end-1))+sum(log(factorial(Cases)))
% 
         LLh=-NO*log(pOi)-(nti-NO)*log(1-pOi);
         CaseTot(1:end-1)
        LLc=-Cases'*log(CaseTot(1:end-1))+sum(CaseTot(1:end-1));
        LL=LLh+weight*LLc;

% % Lh=prod(nchoosek(nti,NO).*(pOi.^NO).*((1-pOi).^(nti-NO)));
% Lh=-log(prod(binopdf(NO,nti,pOi')))
% Lc=-log(prod(poisspdf(Cases,CaseTot(1:end-1))))
% 
% 
%  LL=Lh+weight*Lc;

    end

options = optimoptions('fmincon','Display','iter','Algorithm','sqp','MaxIter',5);
param0=[bi0 bw0 chi0 alpha0 I10g];
lb=param0*.1;
ub=param0*10;
lb(5)=I10g/2;
ub(5)=I10g*2;
paramfit = fmincon(@negloglike,param0,[],[],[],[],lb,ub,[],options)
   bi=paramfit(1)
         bw=paramfit(2)
          chi=paramfit(3)
           %alpha=paramfit(4)
I10=paramfit(5);
           X(2)=I10;
           X(3)=.0001*X(2);
X(6)=xi/nu*X(2);
X(7)=xi/nu*X(3);
% bi=param0(1)
%          bw=param0(2)
%           chi=param0(3)
%            alpha=param0(4)
% I10=param0(5);
%            X(2)=I10;
%            X(3)=.0001*X(2);
% X(6)=xi/nu*X(2);
% X(7)=xi/nu*X(3);
        
[T1, Z1] = ode45(@(t,x)myoden(t,x,bi,bw,chi),Tspan(1:tcon),X,options); 
    bi2=0.825*bi;
    bw2=0.825*bw;
%       bi2=bi;
%     bw2=bw;
[T2, Z2] = ode45(@(t,x)myoden(t,x,bi2,bw2,chi),Tspan(tcon:end),Z1(end,:),options); 
    Z=[Z1;Z2(2:end,:)];
    T=[T1;T2(2:end)];
 for i=1:Md-1
        tspm=Tspan((i-1)*30+1:i*30);
        cm=interp1(tsp,Ps,tspm)-mean(Ps);
    cm=cm'./max(abs(Ps-mean(Ps)));
%     size(cm)
%     size(Z(i:i+30,5))
      
        CaseMonO(i)=sum(gamma/30*Z((i-1)*30+1:i*30,2))/4*N;
        CaseMonI(i)=sum(gamma/30*Z((i-1)*30+1:i*30,3))/4*N;
      
       
        CaseTot(i)=(CaseMonO(i)+CaseMonI(i));
        if i<=Nd
            pOi(i)=CaseMonO(i)/(CaseTot(i));
            
        end
      
   end

figure(1)
plot(T,(Z(:,2)+Z(:,3))/4*N)

figure(2)
plot(T,Z(:,2)*N/4,'b-',T,Z(:,3)*N/4,'r-')

figure(3)
plot(1:Md-1,CaseTot,1:Md-2,Cases,'*')

figure(4)
plot(1:Nd,pOi,1:Nd,pOd,'*')
end