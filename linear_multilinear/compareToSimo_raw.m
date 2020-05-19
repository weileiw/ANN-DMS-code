addpath('~/DATA');
addpath('~/myfun')
addpath('../datasets/DMS/')
load PMEL_May2020.mat

SAL = data_DMS(:,iSAL);
DMS = data_DMS(:,iDMS);
Chl = data_DMS(:,iChl);
PAR = data_DMS(:,iPAR);
MLD = data_DMS(:,iMLD);
Kd490 = data_DMS(:,iKd490);

% data cleaning
Chl(Chl<=0 | Chl>100) = nan;
MLD(MLD<0) = nan;

% calculate solar radiation dose according to VS 2007;

% compare to Simo and Dachs 2002 GBC using in situ Chla
DMS_Chl_MLD = [DMS, MLD, Chl];
g = find(~isnan(sum(DMS_Chl_MLD,2)));

DMS_keep = DMS_Chl_MLD(g, 1);
MLD_keep = DMS_Chl_MLD(g, 2);
Chl_keep = DMS_Chl_MLD(g, 3);

%%%%%%%%%%%%%%%%%%%%%%%%% test with original SD02 parameters#######
testparam.sd02 = []; % use default parameters
[dms_out.sd02,sdcrit] = dms_sd02(Chl_keep,MLD_keep,testparam.sd02);

fprintf('R^2 between obs and SD02 is  %2.2e \n\n',...
                  rsquare(DMS_keep,dms_out.sd02));
%%%%%%%%%%%%%%%%%%%%%%%%% test with original SD02 parameters#######

%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$
% calculate Chl to MLD ratio
Rc2d = Chl_keep ./ MLD_keep;

% find index for Chl/MLD < 0.02;
ismall = find(Rc2d < 0.02);
p = polyfit(log(MLD_keep(ismall)), DMS_keep(ismall),1);
yfit = polyval(p,log(MLD_keep(ismall)));
% get R2 for the fit
fprintf('there are %d data points. \n',length(ismall))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for log(MLD) and DMS, R^2 = %2.2e \n\n',...
         rsquare(DMS_keep(ismall),yfit));

subplot(1,3,1)
plot(DMS_keep(ismall),yfit,'ro','MarkerSize',1)
title('log(MLD)')
hold on
plot([min(DMS_keep),max(DMS_keep)],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(DMS_keep) max(DMS_keep)])
ylim([min(yfit) max(yfit)])
hold off

% find index for Chl/MLD >= 0.02;
ibig = find(Rc2d >= 0.02);
p = polyfit(Rc2d(ibig), DMS_keep(ibig),1);
yfit = polyval(p,Rc2d(ibig));
% get R2 for the fit
fprintf('there are %d data points. \n',length(ibig))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for Chl/MLD and DMS, R^2 = %2.2e \n\n',...
        rsquare(DMS_keep(ibig),yfit));

subplot(1,3,2)
plot(DMS_keep(ibig),yfit,'ro','MarkerSize',1)
title('Chl/MLD')
hold on
plot([min(DMS_keep),max(DMS_keep)],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(DMS_keep) max(DMS_keep)])
ylim([min(yfit) max(yfit)])
hold off
%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$

% compare to Vallina and Simo 2007;
DMS_PAR_Kd = [DMS, PAR, Kd490, MLD];
g = find(~isnan(sum(DMS_PAR_Kd,2)));
DMS_keep = DMS_PAR_Kd(g,1);
PAR_keep = DMS_PAR_Kd(g,2);
Kd490_keep = DMS_PAR_Kd(g,3);
MLD_keep = DMS_PAR_Kd(g,4);
%%%%%%%%%%%%%%%%%%%%%%%%% test with original SD02 parameters#######
testparam.vs07 = []; % use default parameters
dms_out.vs07 = dms_vs07(PAR_keep,MLD_keep,Kd490_keep,testparam.vs07);
fprintf('R^2 between obs and VS07 is  %2.2e \n\n',...
                  rsquare(DMS_keep,dms_out.vs07));
%%%%%%%%%%%%%%%%%%%%%%%%% test with original SD02 parameters#######

%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$
[~,SRD_keep] = dms_vs07(PAR_keep,MLD_keep,Kd490_keep,testparam.vs07);

p = polyfit(SRD_keep, DMS_keep,1);
yfit = polyval(p,SRD_keep);
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SRD and DMS, R^2 = %2.2e \n\n',...
        rsquare(DMS_keep,yfit));

subplot(1,3,3)
plot(DMS_keep,yfit,'ro','MarkerSize',1);
title('SRD')
hold on
plot([min(DMS_keep),max(DMS_keep)],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(DMS_keep) max(DMS_keep)])
ylim([min(yfit) max(yfit)])
hold off
%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$
