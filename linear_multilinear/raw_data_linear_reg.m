3% read online PMEL DMS data.
clear all; close all; clc;
addpath('~/myfun')
% load NAMMES data
load ../datasets/DMS/NAMMES_May2020.mat
Lat_NAMMES = data_DMS(:,iLat);
Lon_NAMMES = data_DMS(:,iLon);
DMS_NAMMES = data_DMS(:,iDMS);
Chl_NAMMES = data_DMS(:,iChl);  
SST_NAMMES = data_DMS(:,iSST);
SAL_NAMMES = data_DMS(:,iSAL);

% load PMEL data
data = readtable('../datasets/DMS/dms_d9t267.dat.txt', 'HeaderLines', 9);
ID = table2array(data(:,'ContributionNumber'));
DateTime = table2array(data(:,'DateTime'));
Lat = table2array(data(:,'Lat'));
Lon = wrapTo360(table2array(data(:,'Lon')));
DMS = table2array(data(:,'swDMS'));
DMSPaq = table2array(data(:,'DMSPaq'));
DMSPp  = table2array(data(:,'DMSPp'));
DMSPt  = table2array(data(:,'DMSPt'));
wdepth = table2array(data(:,'wdepth'));
sdepth = table2array(data(:,'sdepth'));
Chl    = table2array(data(:,'chl'));
SST    = table2array(data(:,'SST'));
SAL    = table2array(data(:,'SAL'));
wspd   = table2array(data(:,'wspd'));
flag   = table2array(data(:,'flag'));
[Year,Month] = datevec(DateTime);

% convert cell to double
DMSPt = str2double(DMSPt);

ikeep = find(DMS > 0.1 & DMS < 100 & sdepth < 10 & SAL > 30);
DMS = DMS(ikeep);
SAL = SAL(ikeep);
SST = SST(ikeep);
DMSPt = DMSPt(ikeep);

data_DMS = [DMS, SAL, SST];
iDMS = 1;
iSAL = 2;
iSST = 3;

DMS = data_DMS(:,iDMS);
tmp =log(DMS);
data_DMS(:,iDMS)= (tmp-nanmean(tmp))/nanstd(tmp);

SAL = data_DMS(:,iSAL);
SAL(SAL<30) = nan;
tmp = log(SAL);
data_DMS(:,iSAL)=(tmp-nanmean(tmp))/nanstd(tmp);

SST = data_DMS(:,iSST)+273.15;
tmp = log(SST);
data_DMS(:,iSST)=(tmp-nanmean(tmp))/nanstd(tmp);

% regression for SAL
g = find(isnan(sum(data_DMS(:,[iDMS iSAL]),2))==0);
p = polyfit(data_DMS(g,iSAL),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSAL));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SAL and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));

% regression for SST
g = find(isnan(sum(data_DMS(:,[iDMS iSST]),2))==0);
p = polyfit(data_DMS(g,iSST),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSST));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SST and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));

fprintf('-------------------------NOW COMBINED DATA ------------------------\n')
% combine PMEL and NAMMES data
ikeep = find(DMS > 0.1 & DMS < 100);
DMS = DMS(ikeep);
SAL = SAL(ikeep);
SST = SST(ikeep);

DMS = [DMS; DMS_NAMMES];
SST = [SST; SST_NAMMES];
SAL = [SAL; SAL_NAMMES];


data_DMS = [DMS, SAL, SST];
iDMS = 1;
iSAL = 2;
iSST = 3;

SAL = data_DMS(:,iSAL);
SAL(SAL<30) = nan;
tmp = log(SAL);
data_DMS(:,iSAL)=(tmp-nanmean(tmp))/nanstd(tmp);

DMS = data_DMS(:,iDMS);
tmp =log(DMS);
data_DMS(:,iDMS)= (tmp-nanmean(tmp))/nanstd(tmp);

SST = data_DMS(:,iSST)+273.15;
tmp = log(SST);
data_DMS(:,iSST)=(tmp-nanmean(tmp))/nanstd(tmp);

% regression for SAL
g = find(isnan(sum(data_DMS(:,[iDMS iSAL]),2))==0);
p = polyfit(data_DMS(g,iSAL),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSAL));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SAL and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));

% regression for SST
g = find(isnan(sum(data_DMS(:,[iDMS iSST]),2))==0);
p = polyfit(data_DMS(g,iSST),data_DMS(g,iDMS),1);
yfit = polyval(p,data_DMS(g,iSST));
% get R2 for the fit
fprintf('there are %d data points. \n',size(g,1))
fprintf('slope is %3.3f. \n',p(1))
fprintf('1st order linear regression fit for SST and DMS, R^2 = %2.2e \n\n',...
        rsquare(data_DMS(g,iDMS),yfit));
