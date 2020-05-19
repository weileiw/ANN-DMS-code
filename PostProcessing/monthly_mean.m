clc; close all; clear all
addpath('~/DATA')
addpath('~/myfun')
addpath('OutPut_data/CESM_filled')

Y = [-89.5:1:89.5]; X = [1:1:360];
[Xq,Yq] = meshgrid(X,Y);

XT = cos(degtorad(Yq)) * 111321;
% Each degree of latitude is about 111 kilometers apart
YT = 111000*ones(180,360);

Area = XT.*YT;

load Lana_climatology/DMS_climate_lana_180x360.mat
for month = 1:12
    tmp = DMS(:,:,month);
    ikeep = find(tmp(:)>0);
    mean_lana(month) = sum(tmp(ikeep).*Area(ikeep))/sum(Area(ikeep));
end

Files=dir('*.*');
for k=1:length(Files)
    FileNames=Files(k).name
end

seed = [2,4,8,16,32,64,128,256,512,1024];
for kk = 1:length(seed)
    file_name = sprintf("mean_seed%d",seed(kk));
    load(file_name); % ave
    A = ave;
    for jj = 1:12
        dat = reshape(A(:,jj),[360,180])';
        ikeep = find(dat(:)>0);
        Mean_mine(jj,kk) = sum(dat(ikeep).*Area(ikeep))/sum(Area(ikeep));
        Anuall(:,:,jj,kk) = dat;
    end    
end
std_mine = std(Mean_mine');
mean_mine = mean(Mean_mine,2)';

mean_SLW = [2.47 2.32 2.26 2.30 2.44 2.05 1.81 1.82 2.04 2.31 2.52 2.70];

scatter([1:12],mean_mine);
hold on
P1 = errorbar([1:12],mean_mine,std_mine,'ro-','MarkerSize',8,'LineWidth',2);
hold on
P2 = plot(mean_lana,'b--*','MarkerSize',8,'LineWidth',2);
% hold on
% P3 = plot(mean_SLW,'k: >');
legend([P1,P2],'This study','Lana et al. 2011')
hold off
xlim([1 12])
xticks([1:12])
xticklabels({'Jan','Feb','Mar','Apr','May','Jun','Jul',...
            'Aug','Sep','Oct','Nov','Dec'})
xlabel('Month')
ylabel('Monthly mean DMS concentration (nM)')

exportfig(gcf,'monthly_mean','fontsize',12,'fontmode','fixed','color','rgb','renderer','painters')

for jk = 1:12
    Mine_ann(:,:,jk) = nanmean(squeeze(Anuall(:,:,jk,:)),3);
end

Mine_ann = nanmean(Mine_ann,3);
Lane_ann = nanmean(DMS,3);

for ji = 1:180
    tmp1 = Mine_ann(ji,:);
    tmp2 = Lane_ann(ji,:);
    tmpArea = Area(ji,:);
    ikeep1 = find(tmp1>0);
    ikeep2 = find(tmp2>0);
    zonal_mine(ji) = nansum(tmp1(ikeep1).*tmpArea(ikeep1))/nansum(tmpArea(ikeep1)); 
    zonal_lana(ji) = nansum(tmp2(ikeep2).*tmpArea(ikeep2))/nansum(tmpArea(ikeep2));
end