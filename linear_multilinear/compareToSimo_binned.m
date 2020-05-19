clc; clear all; close all;
addpath('~/DATA');
addpath('../datasets/DMS/')
load PMEL_DMS_180x360.mat DMS Chl
% load PMEL_NAMMES_DMS_180x360.mat
load MIMOC_MLD_180x360x12.mat
% load Chl_180x360x12.mat Chl
load PAR_180x360x12.mat PAR
load KD_180x360x12.mat  Kd490

% calculate solar radiation dose according to VS 2007;

DMS_all = [];
PAR_all = [];
MLD_all = [];
Chl_all = [];
Kd490_all = [];
for jj = 1:12
    DMS_tmp = DMS(:,:,jj);
    PAR_tmp = PAR(:,:,jj);
    MLD_tmp = MLD(:,:,jj);
    Chl_tmp = Chl(:,:,jj);
    Kd_tmp  = Kd490(:,:,jj);
    
    ikeep = find(DMS_tmp(:)>0);
    DMS_sub = DMS_tmp(ikeep);
    PAR_sub = PAR_tmp(ikeep);
    MLD_sub = MLD_tmp(ikeep);
    Chl_sub = Chl_tmp(ikeep);
    Kd_sub  = Kd_tmp(ikeep);

    DMS_all = [DMS_all;DMS_sub];
    PAR_all = [PAR_all;PAR_sub];
    MLD_all = [MLD_all;MLD_sub];
    Chl_all = [Chl_all;Chl_sub];
    Kd490_all = [Kd490_all;Kd_sub];
end

% compare to Simo and Dachs 2002 GBC
DMS_Chl_MLD = [DMS_all, MLD_all, Chl_all];
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

% find index for Chl/MLD >= 0.02;
ibig = find(Rc2d >= 0.02);
p = polyfit(Rc2d(ibig), DMS_keep(ibig),1);
yfit = polyval(p,Rc2d(ibig));
% get R2 for the fit
fprintf('there are %d data points. \n',length(ibig))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for Chl/MLD and DMS, R^2 = %2.2e \n\n',...
        rsquare(DMS_keep(ibig),yfit));
%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$


% compare to Vallina and Simo 2007;
DMS_PAR_Kd = [DMS_all, PAR_all, Kd490_all, MLD_all];
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
%%%%%%%%%%%%%%%%%%%%%%% Redo regression $$$$$$$$$$$$$$$$$$$$$$$$$$$$

