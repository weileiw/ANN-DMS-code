clc; close all; clear all
addpath('~/myfun/')
clearvars;
dirr = '..'; source = [dirr,'/datasets/DMS'];
path(path,source);
version={'surf_'};

save_path=[dirr,'/OUTPUT'];
% load PMEL_May2020.mat
% load NAMMES_May2020.mat
load PMEL_NAMMES_May2020.mat

%Variable limits and transformation dec to log
SAL = data_DMS(:,iSAL);
tmp = log(SAL);
data_DMS(:,iSAL)=(tmp-nanmean(tmp))/nanstd(tmp);

DMS = data_DMS(:,iDMS);
tmp =log(DMS);
data_DMS(:,iDMS)= (tmp-nanmean(tmp))/nanstd(tmp);

DMSP = data_DMS(:,iDMSPt);
DMSP(DMSP < 1) = nan;
tmp =log(DMSP);
data_DMS(:,iDMSPt)= (tmp-nanmean(tmp))/nanstd(tmp);

% in situ Chl a
Chl = data_DMS(:,iChl);
Chl(Chl <= 0.01 | Chl > 20) = nan;
tmp =log(Chl);
data_DMS(:,iChl)=(tmp-nanmean(tmp))/nanstd(tmp);

% satellite Chl a
Chl_sat = data_DMS(:,iChl_sat);
Chl_sat(Chl_sat <= 0.01 | Chl_sat>20) = nan;
tmp =log(Chl_sat);
data_DMS(:,iChl_sat)=(tmp-nanmean(tmp))/nanstd(tmp);

% satellite POC
POC = data_DMS(:,iPOC);
pc = prctile(POC,[0.5 99.5]);
POC(POC <= pc(1)) = nan;
POC(POC >= pc(2)) = nan;
tmp =log(POC);
data_DMS(:,iPOC)=(tmp-nanmean(tmp))/nanstd(tmp);

% satellite PIC
PIC = data_DMS(:,iPIC);
pc = prctile(PIC,[0.5 99.5]);
PIC(PIC <= pc(1)) = nan;
PIC(PIC >= pc(2)) = nan;
tmp =log(PIC);
data_DMS(:,iPIC)=(tmp-nanmean(tmp))/nanstd(tmp);

%Filter: use only deep samples
PAR = data_DMS(:,iPAR);
PAR(PAR <= 0) = nan;
tmp = log(PAR);
data_DMS(:,iPAR)=(tmp-nanmean(tmp))/nanstd(tmp);

MLD = data_DMS(:,iMLD);
MLD(MLD <= 0 | MLD>150) = nan;
tmp = log(MLD);
data_DMS(:,iMLD)=(tmp-nanmean(tmp))/nanstd(tmp);

SiO = data_DMS(:,iSiO);
SiO(SiO <= 0.1) = nan;
tmp = log(SiO);
data_DMS(:,iSiO)=(tmp-nanmean(tmp))/nanstd(tmp);

PO4 = data_DMS(:,iPO4);
PO4(PO4 <= 1e-2) = nan;
tmp = log(PO4);
data_DMS(:,iPO4)=(tmp-nanmean(tmp))/nanstd(tmp);

NO3 = data_DMS(:,iNO3);
NO3(NO3 <= 1e-2) = nan;
tmp = log(NO3);
data_DMS(:,iNO3)=(tmp-nanmean(tmp))/nanstd(tmp);


SST = data_DMS(:,iSST)+273.15;
tmp = log(SST);
data_DMS(:,iSST)=(tmp-nanmean(tmp))/nanstd(tmp);

figure(1)
% regression for PAR
g = find(isnan(sum(data_DMS(:,[iDMS iPAR]),2))==0);
p = polyfit(data_DMS(g,iPAR),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iPAR));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for PAR and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,1)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('PAR')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for MLD
g = find(isnan(sum(data_DMS(:,[iDMS iMLD]),2))==0);
p = polyfit(data_DMS(g,iMLD),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iMLD));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for MLD and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,2)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('MLD')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for Chl
g = find(isnan(sum(data_DMS(:,[iDMS iChl]),2))==0);
p = polyfit(data_DMS(g,iChl),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iChl));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf(['1st order linear regression fit for in situ Chl and DMS, ' ...
         'R^2 = %2.2e \n\n'], rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,3)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('in situ Chl')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for Chl
g = find(isnan(sum(data_DMS(:,[iDMS iChl_sat]),2))==0);
p = polyfit(data_DMS(g,iChl_sat),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iChl_sat));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf(['1st order linear regression fit for satellite Chl and DMS, ' ...
         'R^2 = %2.2e \n\n'], rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,4)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('satellite Chl')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for SAL
g = find(isnan(sum(data_DMS(:,[iDMS iSAL]),2))==0);
p = polyfit(data_DMS(g,iSAL),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSAL));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SAL and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,5)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('SAL')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for SST
g = find(isnan(sum(data_DMS(:,[iDMS iSST]),2))==0);
p = polyfit(data_DMS(g,iSST),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSST));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SST and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,6)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('SST')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

figure(2)
% regression for SiO
g = find(isnan(sum(data_DMS(:,[iDMS iSiO]),2))==0);
p = polyfit(data_DMS(g,iSiO),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSiO));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SiO and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,1)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('Salicate')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for PO4
g = find(isnan(sum(data_DMS(:,[iDMS iPO4]),2))==0);
p = polyfit(data_DMS(g,iPO4),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iPO4));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for PO4 and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,2)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('Phosphate')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for NO3
g = find(isnan(sum(data_DMS(:,[iDMS iNO3]),2))==0);
p = polyfit(data_DMS(g,iNO3),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iNO3));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for NO3 and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,3)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('Nitrate')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% regression for DMSPt
g = find(isnan(sum(data_DMS(:,[iDMS iDMSPt]),2))==0);
if length(g) > 0
    p = polyfit(data_DMS(g,iDMSPt),data_DMS(g,iDMS),1);
    yfit = polyval(p,data_DMS(g,iDMSPt));
    % get R2 for the fit
    fprintf('there are %d data points. \n',size(g,1))
    fprintf('slope is %3.3f. \n',p(1))
    fprintf('1st order linear regression fit for DMSP and DMS, R^2 = %2.2e \n\n',...
            rsquare(data_DMS(g,iDMS),yfit));
    subplot(2,3,4)
    plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
    title('DMSPt')
    hold on
    plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
         [min(data_DMS(g,iDMS)), max(data_DMS(g,iDMS))],'b.-')
    xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
    ylim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
    hold off
end

% regression for POC
g = find(isnan(sum(data_DMS(:,[iDMS iPOC]),2))==0);
p = polyfit(data_DMS(g,iPOC),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iPOC));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for POC and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,5)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('POC')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(data_DMS(g,iDMS)), max(data_DMS(g,iDMS))],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
hold off

% regression for PIC
g = find(isnan(sum(data_DMS(:,[iDMS iPIC]),2))==0);
p = polyfit(data_DMS(g,iPIC),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iPIC));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for PIC and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
subplot(2,3,6)
plot(data_DMS(g,iDMS),yfit,'ro','MarkerSize',1)
title('PIC')
hold on
plot([min(data_DMS(g,iDMS)),max(data_DMS(g,iDMS))],...
     [min(yfit), max(yfit)],'b.-')
xlim([min(data_DMS(g,iDMS)) max(data_DMS(g,iDMS))])
ylim([min(yfit) max(yfit)])
hold off

% exportfig(gcf,'OUTPUT/linear_NAMMES','fontsize',12,'fontmode', ...
          % 'fixed','color','rgb','renderer','painters')
