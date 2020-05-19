clc; close all; clear all
addpath('./output_data');
addpath('~/Dropbox/myfunc')
load ../Lana_Gali_VS_SD/Gali_climat.mat
load ../Lana_Gali_VS_SD/DMS_climate_lana_180x360.mat
monthly_mean_L10 = DMS;
clear DMS

Files=dir('./output_data/*.mat');
for k=1:length(Files)
    FileNames=Files(k).name;
    DMS_tmp = load(FileNames);
    DMS(:,:,:,k) = DMS_tmp.DMS;
end

monthly_mean_W20 = squeeze(nanmean(DMS,4));
% DMS = monthly_mean_W20;
% save monthly_climatology_W20 DMS
% clear DMS 


Y = [-89.5:1:89.5]; X = [1:1:360];
[Xq,Yq] = meshgrid(X,Y);

XT = cos(degtorad(Yq)) * 111321;
% Each degree of latitude is about 111 kilometers apart
YT = 111000*ones(180,360);

Area = XT.*YT;

Mine_spring = nanmean(monthly_mean_W20(:,:,[3:5]),3);
Lane_spring = nanmean(monthly_mean_L10(:,:,[3:5]),3);
SD02_spring = nanmean(SD02(:,:,[3:5]),3);
VS07_spring = nanmean(VS07(:,:,[3:5]),3);
GSM_KD_spring = nanmean(GSM_KD(:,:,[3:5]),3);
CHL_KD_spring = nanmean(CHL_KD(:,:,[3:5]),3);
GSM_ZLEE_spring = nanmean(GSM_ZLEE(:,:,[3:5]),3);
CHL_ZLEE_spring = nanmean(CHL_ZLEE(:,:,[3:5]),3);

for ji = 1:180
    tmp1 = Mine_spring(ji,:);
    tmp2 = Lane_spring(ji,:);
    tmp3 = SD02_spring(ji,:);
    tmp4 = VS07_spring(ji,:);
    tmp5 = GSM_KD_spring(ji,:);
    tmp6 = CHL_KD_spring(ji,:);
    tmp7 = GSM_ZLEE_spring(ji,:);
    tmp8 = CHL_ZLEE_spring(ji,:);
    
    tmpArea = Area(ji,:);
    ikeep1 = find(tmp1>0);
    ikeep2 = find(tmp2>0);
    ikeep3 = find(tmp3>0);
    ikeep4 = find(tmp4>0);
    ikeep5 = find(tmp5>0);
    ikeep6 = find(tmp6>0);
    ikeep7 = find(tmp7>0);
    ikeep8 = find(tmp8>0);
    zonal_W20(ji) = nansum(tmp1(ikeep1).*tmpArea(ikeep1))/nansum(tmpArea(ikeep1)); 
    zonal_lana(ji) = nansum(tmp2(ikeep2).*tmpArea(ikeep2))/nansum(tmpArea(ikeep2));
    zonal_SD02(ji) = nansum(tmp3(ikeep3).*tmpArea(ikeep3))/nansum(tmpArea(ikeep3)); 
    zonal_VS07(ji) = nansum(tmp4(ikeep4).*tmpArea(ikeep4))/nansum(tmpArea(ikeep4));
    zonal_GSM_KD(ji) = nansum(tmp5(ikeep5).*tmpArea(ikeep5))/nansum(tmpArea(ikeep5)); 
    zonal_CHL_KD(ji) = nansum(tmp6(ikeep6).*tmpArea(ikeep6))/nansum(tmpArea(ikeep6));
    zonal_GSM_ZLEE(ji) = nansum(tmp7(ikeep7).*tmpArea(ikeep7))/nansum(tmpArea(ikeep7)); 
    zonal_CHL_ZLEE(ji) = nansum(tmp8(ikeep8).*tmpArea(ikeep8))/nansum(tmpArea(ikeep8));
end

P1 = plot(zonal_W20,'ro-','MarkerSize',4,'LineWidth',2);
hold on
P2 = plot(zonal_lana,'g:*','MarkerSize',4,'LineWidth',2);
hold on
P3 = plot(zonal_VS07,'b:+','MarkerSize',4,'LineWidth',2);
hold on
P4 = plot(zonal_SD02,'c:.','MarkerSize',4,'LineWidth',2);
hold on
P5 = plot(zonal_GSM_KD,'k:x','MarkerSize',4,'LineWidth',2);
hold on
P6 = plot(zonal_CHL_KD,'m:x','MarkerSize',4,'LineWidth',2);
hold on
P7 = plot(zonal_GSM_ZLEE,'k:^','MarkerSize',4,'LineWidth',2);
hold on
P8 = plot(zonal_CHL_ZLEE,'m:^','MarkerSize',4,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],{'Thisstudy','L11','VS07',...
                    'SD02','GSM-KD','CHL-KD','GSM-ZLEE','CHL-ZLEE'})
hold off
xlim([1 181])
ylim([0 12])
xticks([1:30:181])
xticklabels({'90S','60S','30S','Equator','30N','60N','90N'})
xlabel('Latitude')
ylabel('Zonal mean DMS concentration (nM)')
    
exportfig(gcf,'./FIGs/zonal_mean_spring','fontsize',12,'fontmode','fixed',...
          'color','rgb','renderer','painters')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mine_summer = nanmean(monthly_mean_W20(:,:,[6:8]),3);
Lane_summer = nanmean(monthly_mean_L10(:,:,[6:8]),3);
SD02_summer = nanmean(SD02(:,:,[6:8]),3);
VS07_summer = nanmean(VS07(:,:,[6:8]),3);
GSM_KD_summer = nanmean(GSM_KD(:,:,[6:8]),3);
CHL_KD_summer = nanmean(CHL_KD(:,:,[6:8]),3);
GSM_ZLEE_summer = nanmean(GSM_ZLEE(:,:,[6:8]),3);
CHL_ZLEE_summer = nanmean(CHL_ZLEE(:,:,[6:8]),3);

for ji = 1:180
    tmp1 = Mine_summer(ji,:);
    tmp2 = Lane_summer(ji,:);
    tmp3 = SD02_summer(ji,:);
    tmp4 = VS07_summer(ji,:);
    tmp5 = GSM_KD_summer(ji,:);
    tmp6 = CHL_KD_summer(ji,:);
    tmp7 = GSM_ZLEE_summer(ji,:);
    tmp8 = CHL_ZLEE_summer(ji,:);
    
    tmpArea = Area(ji,:);
    ikeep1 = find(tmp1>0);
    ikeep2 = find(tmp2>0);
    ikeep3 = find(tmp3>0);
    ikeep4 = find(tmp4>0);
    ikeep5 = find(tmp5>0);
    ikeep6 = find(tmp6>0);
    ikeep7 = find(tmp7>0);
    ikeep8 = find(tmp8>0);
    zonal_W20(ji) = nansum(tmp1(ikeep1).*tmpArea(ikeep1))/nansum(tmpArea(ikeep1)); 
    zonal_lana(ji) = nansum(tmp2(ikeep2).*tmpArea(ikeep2))/nansum(tmpArea(ikeep2));
    zonal_SD02(ji) = nansum(tmp3(ikeep3).*tmpArea(ikeep3))/nansum(tmpArea(ikeep3)); 
    zonal_VS07(ji) = nansum(tmp4(ikeep4).*tmpArea(ikeep4))/nansum(tmpArea(ikeep4));
    zonal_GSM_KD(ji) = nansum(tmp5(ikeep5).*tmpArea(ikeep5))/nansum(tmpArea(ikeep5)); 
    zonal_CHL_KD(ji) = nansum(tmp6(ikeep6).*tmpArea(ikeep6))/nansum(tmpArea(ikeep6));
    zonal_GSM_ZLEE(ji) = nansum(tmp7(ikeep7).*tmpArea(ikeep7))/nansum(tmpArea(ikeep7)); 
    zonal_CHL_ZLEE(ji) = nansum(tmp8(ikeep8).*tmpArea(ikeep8))/nansum(tmpArea(ikeep8));
end

P1 = plot(zonal_W20,'ro-','MarkerSize',4,'LineWidth',2);
hold on
P2 = plot(zonal_lana,'g:*','MarkerSize',4,'LineWidth',2);
hold on
P3 = plot(zonal_VS07,'b:+','MarkerSize',4,'LineWidth',2);
hold on
P4 = plot(zonal_SD02,'c:.','MarkerSize',4,'LineWidth',2);
hold on
P5 = plot(zonal_GSM_KD,'k:x','MarkerSize',4,'LineWidth',2);
hold on
P6 = plot(zonal_CHL_KD,'m:x','MarkerSize',4,'LineWidth',2);
hold on
P7 = plot(zonal_GSM_ZLEE,'k:^','MarkerSize',4,'LineWidth',2);
hold on
P8 = plot(zonal_CHL_ZLEE,'m:^','MarkerSize',4,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],{'Thisstudy','L11','VS07',...
                    'SD02','GSM_KD','CHL-KD','GSM-ZLEE','CHL-ZLEE'})
hold off
xlim([1 181])
ylim([0 12])
xticks([1:30:181])
xticklabels({'90S','60S','30S','Equator','30N','60N','90N'})
xlabel('Latitude')
ylabel('Zonal mean DMS concentration (nM)')
    
exportfig(gcf,'./FIGs/zonal_mean_summer','fontsize',12,'fontmode','fixed',...
          'color','rgb','renderer','painters')


Mine_fall = nanmean(monthly_mean_W20(:,:,[9:11]),3);
Lane_fall = nanmean(monthly_mean_L10(:,:,[9:11]),3);
SD02_fall = nanmean(SD02(:,:,[9:11]),3);
VS07_fall = nanmean(VS07(:,:,[9:11]),3);
GSM_KD_fall = nanmean(GSM_KD(:,:,[9:11]),3);
CHL_KD_fall = nanmean(CHL_KD(:,:,[9:11]),3);
GSM_ZLEE_fall = nanmean(GSM_ZLEE(:,:,[9:11]),3);
CHL_ZLEE_fall = nanmean(CHL_ZLEE(:,:,[9:11]),3);

for ji = 1:180
    tmp1 = Mine_fall(ji,:);
    tmp2 = Lane_fall(ji,:);
    tmp3 = SD02_fall(ji,:);
    tmp4 = VS07_fall(ji,:);
    tmp5 = GSM_KD_fall(ji,:);
    tmp6 = CHL_KD_fall(ji,:);
    tmp7 = GSM_ZLEE_fall(ji,:);
    tmp8 = CHL_ZLEE_fall(ji,:);
    
    tmpArea = Area(ji,:);
    ikeep1 = find(tmp1>0);
    ikeep2 = find(tmp2>0);
    ikeep3 = find(tmp3>0);
    ikeep4 = find(tmp4>0);
    ikeep5 = find(tmp5>0);
    ikeep6 = find(tmp6>0);
    ikeep7 = find(tmp7>0);
    ikeep8 = find(tmp8>0);
    zonal_W20(ji) = nansum(tmp1(ikeep1).*tmpArea(ikeep1))/nansum(tmpArea(ikeep1)); 
    zonal_lana(ji) = nansum(tmp2(ikeep2).*tmpArea(ikeep2))/nansum(tmpArea(ikeep2));
    zonal_SD02(ji) = nansum(tmp3(ikeep3).*tmpArea(ikeep3))/nansum(tmpArea(ikeep3)); 
    zonal_VS07(ji) = nansum(tmp4(ikeep4).*tmpArea(ikeep4))/nansum(tmpArea(ikeep4));
    zonal_GSM_KD(ji) = nansum(tmp5(ikeep5).*tmpArea(ikeep5))/nansum(tmpArea(ikeep5)); 
    zonal_CHL_KD(ji) = nansum(tmp6(ikeep6).*tmpArea(ikeep6))/nansum(tmpArea(ikeep6));
    zonal_GSM_ZLEE(ji) = nansum(tmp7(ikeep7).*tmpArea(ikeep7))/nansum(tmpArea(ikeep7)); 
    zonal_CHL_ZLEE(ji) = nansum(tmp8(ikeep8).*tmpArea(ikeep8))/nansum(tmpArea(ikeep8));
end

P1 = plot(zonal_W20,'ro-','MarkerSize',4,'LineWidth',2);
hold on
P2 = plot(zonal_lana,'g:*','MarkerSize',4,'LineWidth',2);
hold on
P3 = plot(zonal_VS07,'b:+','MarkerSize',4,'LineWidth',2);
hold on
P4 = plot(zonal_SD02,'c:.','MarkerSize',4,'LineWidth',2);
hold on
P5 = plot(zonal_GSM_KD,'k:x','MarkerSize',4,'LineWidth',2);
hold on
P6 = plot(zonal_CHL_KD,'m:x','MarkerSize',4,'LineWidth',2);
hold on
P7 = plot(zonal_GSM_ZLEE,'k:^','MarkerSize',4,'LineWidth',2);
hold on
P8 = plot(zonal_CHL_ZLEE,'m:^','MarkerSize',4,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],{'Thisstudy','L11','VS07',...
                    'SD02','GSM-KD','CHL-KD','GSM-ZLEE','CHL-ZLEE'})
hold off
xlim([1 181])
ylim([0 12])
xticks([1:30:181])
xticklabels({'90S','60S','30S','Equator','30N','60N','90N'})
xlabel('Latitude')
ylabel('Zonal mean DMS concentration (nM)')
    
exportfig(gcf,'./FIGs/zonal_mean_fall','fontsize',12,'fontmode','fixed',...
          'color','rgb','renderer','painters')


Mine_winter = nanmean(monthly_mean_W20(:,:,[12,1:2]),3);
Lane_winter = nanmean(monthly_mean_L10(:,:,[12,1:2]),3);
SD02_winter = nanmean(SD02(:,:,[12,1:2]),3);
VS07_winter = nanmean(VS07(:,:,[12,1:2]),3);
GSM_KD_winter = nanmean(GSM_KD(:,:,[12,1:2]),3);
CHL_KD_winter = nanmean(CHL_KD(:,:,[12,1:2]),3);
GSM_ZLEE_winter = nanmean(GSM_ZLEE(:,:,[12,1:2]),3);
CHL_ZLEE_winter = nanmean(CHL_ZLEE(:,:,[12,1:2]),3);

for ji = 1:180
    tmp1 = Mine_winter(ji,:);
    tmp2 = Lane_winter(ji,:);
    tmp3 = SD02_winter(ji,:);
    tmp4 = VS07_winter(ji,:);
    tmp5 = GSM_KD_winter(ji,:);
    tmp6 = CHL_KD_winter(ji,:);
    tmp7 = GSM_ZLEE_winter(ji,:);
    tmp8 = CHL_ZLEE_winter(ji,:);
    
    tmpArea = Area(ji,:);
    ikeep1 = find(tmp1>0);
    ikeep2 = find(tmp2>0);
    ikeep3 = find(tmp3>0);
    ikeep4 = find(tmp4>0);
    ikeep5 = find(tmp5>0);
    ikeep6 = find(tmp6>0);
    ikeep7 = find(tmp7>0);
    ikeep8 = find(tmp8>0);
    zonal_W20(ji) = nansum(tmp1(ikeep1).*tmpArea(ikeep1))/nansum(tmpArea(ikeep1)); 
    zonal_lana(ji) = nansum(tmp2(ikeep2).*tmpArea(ikeep2))/nansum(tmpArea(ikeep2));
    zonal_SD02(ji) = nansum(tmp3(ikeep3).*tmpArea(ikeep3))/nansum(tmpArea(ikeep3)); 
    zonal_VS07(ji) = nansum(tmp4(ikeep4).*tmpArea(ikeep4))/nansum(tmpArea(ikeep4));
    zonal_GSM_KD(ji) = nansum(tmp5(ikeep5).*tmpArea(ikeep5))/nansum(tmpArea(ikeep5)); 
    zonal_CHL_KD(ji) = nansum(tmp6(ikeep6).*tmpArea(ikeep6))/nansum(tmpArea(ikeep6));
    zonal_GSM_ZLEE(ji) = nansum(tmp7(ikeep7).*tmpArea(ikeep7))/nansum(tmpArea(ikeep7)); 
    zonal_CHL_ZLEE(ji) = nansum(tmp8(ikeep8).*tmpArea(ikeep8))/nansum(tmpArea(ikeep8));
end

P1 = plot(zonal_W20,'ro-','MarkerSize',4,'LineWidth',2);
hold on
P2 = plot(zonal_lana,'g:*','MarkerSize',4,'LineWidth',2);
hold on
P3 = plot(zonal_VS07,'b:+','MarkerSize',4,'LineWidth',2);
hold on
P4 = plot(zonal_SD02,'c:.','MarkerSize',4,'LineWidth',2);
hold on
P5 = plot(zonal_GSM_KD,'k:x','MarkerSize',4,'LineWidth',2);
hold on
P6 = plot(zonal_CHL_KD,'m:x','MarkerSize',4,'LineWidth',2);
hold on
P7 = plot(zonal_GSM_ZLEE,'k:^','MarkerSize',4,'LineWidth',2);
hold on
P8 = plot(zonal_CHL_ZLEE,'m:^','MarkerSize',4,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],{'Thisstudy','L11','VS07',...
                    'SD02','GSM-KD','CHL-KD','GSM-ZLEE','CHL-ZLEE'})
hold off
xlim([1 181])
ylim([0 12])
xticks([1:30:181])
xticklabels({'90S','60S','30S','Equator','30N','60N','90N'})
xlabel('Latitude')
ylabel('Zonal mean DMS concentration (nM)')
    
exportfig(gcf,'./FIGs/zonal_mean_winter','fontsize',12,'fontmode','fixed',...
          'color','rgb','renderer','painters')
