clc; clear all; close all
addpath('~/myfun/')
clearvars;
dirr = '..'; source = [dirr,'/datasets/DMS'];
path(path,source);
version={'surf_'};

save_path=[dirr,'/OUTPUT'];
load PMEL_NAMMES_May2020.mat

Lat = data_DMS(:,iLat);
Lon = data_DMS(:,iLon);
DOY = data_DMS(:,iDOY);
Time = data_DMS(:,ilocaltime);

latlon1 = sin(Lat * pi / 180);
latlon2 = sin(Lon * pi / 180) .* cos(Lat * pi / 180);
latlon3 =-cos(Lon * pi / 180) .* cos(Lat * pi / 180);
DOY1 = cos(DOY * (2 * pi) / 366);
DOY2 = sin(DOY * (2 * pi) / 366);
Time1 = cos(Time * (2 * pi));
Time2 = sin(Time * (2 * pi));

[nx,ny] = size(data_DMS);
data_DMS(:,ny+1) = latlon1; ilatlon1 = ny+1;
data_DMS(:,ny+2) = latlon2; ilatlon2 = ny+2;
data_DMS(:,ny+3) = latlon3; ilatlon3 = ny+3;
data_DMS(:,ny+4) = DOY1; iDOY1 = ny+4;
data_DMS(:,ny+5) = DOY2; iDOY2 = ny+5;
data_DMS(:,ny+6) = Time1; iTime1 = ny+6;
data_DMS(:,ny+7) = Time2; iTime2 = ny+7;
%Variable limits and transformation dec to log
% not SAL and DMS have been cleaned when data is assembled.
SAL = data_DMS(:,iSAL);
tmp = log(SAL);
data_DMS(:,iSAL)=(tmp-nanmean(tmp))/nanstd(tmp);

DMS = data_DMS(:,iDMS);
tmp =log(DMS);
data_DMS(:,iDMS)= (tmp-nanmean(tmp))/nanstd(tmp);

% satellite Chl a
Chl_sat = data_DMS(:,iChl_sat);
Chl_sat(Chl_sat < 0.01 | Chl_sat > 20) = nan;
tmp =log(Chl_sat);
data_DMS(:,iChl_sat)=(tmp-nanmean(tmp))/nanstd(tmp);

%Filter: use only deep samples
PAR = data_DMS(:,iPAR);
PAR(PAR <= 0) = nan;
tmp = log(PAR);
data_DMS(:,iPAR)=(tmp-nanmean(tmp))/nanstd(tmp);

MLD = data_DMS(:,iMLD);
MLD(MLD > 150) = nan;
tmp = log(MLD);
data_DMS(:,iMLD)=(tmp-nanmean(tmp))/nanstd(tmp);

SiO = data_DMS(:,iSiO);
SiO(SiO < 0.1) = nan;
tmp = log(SiO);
data_DMS(:,iSiO)=(tmp-nanmean(tmp))/nanstd(tmp);

PO4 = data_DMS(:,iPO4);
PO4(PO4 < 1e-2) = nan;
tmp = log(PO4);
data_DMS(:,iPO4)=(tmp-nanmean(tmp))/nanstd(tmp);

NO3 = data_DMS(:,iNO3);
NO3(NO3 < 1e-2) = nan;
tmp = log(NO3);
data_DMS(:,iNO3)=(tmp-nanmean(tmp))/nanstd(tmp);


SST = data_DMS(:,iSST)+273.15;
tmp = log(SST);
data_DMS(:,iSST)=(tmp-nanmean(tmp))/nanstd(tmp);


g = find(isnan(sum(data_DMS(:,[iDMS iPO4 iNO3 iSAL iSST iPAR ...
                    iSiO iMLD iChl_sat]),2))==0); %[z:PBS]
data_DMS = data_DMS(g,:);
fprintf('Total data  points being used are %d \n',length(g));

% env_cols=[iPAR iMLD iChl_sat iSAL iSST iSiO iPO4 iNO3 iPOC iPIC];
env_cols=[iPAR iMLD iChl_sat iSAL iSST iSiO iPO4 iNO3 ilatlon1, ...
          ilatlon2,ilatlon3,iDOY1,iDOY2,iTime1, iTime2];

tbl = data_DMS(:,env_cols);

mdl = stepwiselm(tbl,data_DMS(:,iDMS),'Criterion','BIC','PRemove',0.01);

% function func = make_MLR_function(mdl,tbl)
func = 0;
VA = {'x1' 'x2' 'x3' 'x4' 'x5' 'x6' 'x7' 'x8' 'x9' 'x10' 'x11' 'x12' ...
     'x13' 'x14','x15'};
Var_names = mdl.CoefficientNames;
CoefficientValue = mdl.Coefficients.Estimate;

for jl = 1:length(Var_names)
    if strcmp(Var_names{jl},'(Intercept)')
        func = func + CoefficientValue(jl);

    end
end

for j1 = 1:length(Var_names)
    for j2 = 1:length(VA)
        if strcmp(Var_names{j1},VA{j2})
            func = func + CoefficientValue(j1)*tbl(:,j2);

        end
    end
end

for jl = 1:length(Var_names)
    if strfind(Var_names{jl},':')>0
        tmp = Var_names{jl};
	i1st = extractBefore(tmp,':');
	i2nd = extractAfter(tmp,':');
        for jk = 1:length(VA)
            if  strcmp(VA{jk},i1st);
                data_row1 = jk;
            end
            if  strcmp(VA{jk},i2nd);
                data_row2 = jk;
            end
        end
        func = func + CoefficientValue(jl)* tbl(:,data_row1).*tbl(:,data_row2);

    end
end

for jl = 1:length(Var_names)
    if strfind(Var_names{jl},'^2')
        tmp = Var_names{jl};
        for jm = 1:length(VA)
            if strcmp(VA{jm},tmp(1:2));
                func = func + CoefficientValue(jl)*tbl(:,jm).^2;
            end
        end
    end
end


% plot M. vs O.
cr = 95:-5:5;
W = ones(length(func),1);
data = [data_DMS(:,iDMS),func];
tmp = data(:,1);
[bandwidth,density,X,Y] = ...
    mykde2d(data,100,[-2 -2],[4 4],W);

dx = X(3,5)-X(3,4);
dy = Y(4,2)-Y(3,2);
[q,ii] = sort(density(:)*dx*dy,'descend');
D = density;
D(ii) = cumsum(q);
subplot('position',[0.2 0.2 0.6 0.6])
contourf(X,Y,100*(1-D),cr); hold on
contour(X,Y,100*(1-D),cr);

caxis([5 95])
%set(gca,'FontSize',16);
grid on
axis square
xlabel('observed DMS (nmol/L)');
ylabel('modeled DMS (nmol/L)');
% title('model V.S. observation')
plot([-2 4],[-2 4],'r--','linewidth',2);

subplot('position',[0.82 0.2 0.05 0.6]);
contourf([1 2],cr,[cr(:),cr(:)],cr); hold on
contour([1 2],cr,[cr(:),cr(:)],cr);
hold off
%set(gca,'FontSize',14);
set(gca,'XTickLabel',[]);
set(gca,'YAxisLocation','right');
set(gca,'TickLength',[0 0])
ylabel('(percentile)')

R2 = rsquare(data_DMS(:,iDMS),func)

% exportfig(gcf,'OUTPUT/Multi_PMEL_NAMMES','fontsize',12,'fontmode','fixed','color','rgb','renderer','painters')

