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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for kk = 1:12

    Nmean_W20(kk) = get_mean(monthly_mean_W20(:,:,kk), Area, 'N');
    Nmean_lana(kk) = get_mean(monthly_mean_L10(:,:,kk), Area, 'N');
    Nmean_SD02(kk) = get_mean(SD02(:,:,kk), Area, 'N');
    Nmean_VS07(kk) = get_mean(VS07(:,:,kk), Area, 'N');
    Nmean_GSM_KD(kk) = get_mean(GSM_KD(:,:,kk), Area, 'N');
    Nmean_CHL_KD(kk) = get_mean(CHL_KD(:,:,kk), Area, 'N');
    Nmean_CHL_ZLEE(kk) = get_mean(CHL_ZLEE(:,:,kk), Area, 'N');
    Nmean_GSM_ZLEE(kk) = get_mean(GSM_ZLEE(:,:,kk), Area, 'N');

    Smean_W20(kk) = get_mean(monthly_mean_W20(:,:,kk), Area, 'S');
    Smean_lana(kk) = get_mean(monthly_mean_L10(:,:,kk), Area, 'S');
    Smean_SD02(kk) = get_mean(SD02(:,:,kk), Area, 'S');
    Smean_VS07(kk) = get_mean(VS07(:,:,kk), Area, 'S');
    Smean_GSM_KD(kk) = get_mean(GSM_KD(:,:,kk), Area, 'S');
    Smean_CHL_KD(kk) = get_mean(CHL_KD(:,:,kk), Area, 'S');
    Smean_CHL_ZLEE(kk) = get_mean(CHL_ZLEE(:,:,kk), Area, 'S');
    Smean_GSM_ZLEE(kk) = get_mean(GSM_ZLEE(:,:,kk), Area, 'S');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P1 = plot([1:12],Nmean_W20,'ro-','MarkerSize',8,'LineWidth',2);
hold on
P2 = plot(Nmean_lana,'g:*','MarkerSize',8,'LineWidth',2);
hold on
P3 = plot(Nmean_VS07,'b:+','MarkerSize',8,'LineWidth',2);
hold on
P4 = plot(Nmean_SD02,'c:.','MarkerSize',8,'LineWidth',2);
hold on
P5 = plot(Nmean_GSM_KD,'k:x','MarkerSize',8,'LineWidth',2);
hold on
P6 = plot(Nmean_CHL_KD,'y:x','MarkerSize',8,'LineWidth',2);
hold on
P7 = plot(Nmean_GSM_ZLEE,'k:^','MarkerSize',8,'LineWidth',2);
hold on
P8 = plot(Nmean_CHL_ZLEE,'y:^','MarkerSize',8,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],'This study','L11','VS07','SD02','GSM-KD','CHL-KD','GSM-ZLEE','CHL-ZLEE')
hold off
xlim([0 13])
ylim([0.5 5.5])
xticks([0:13])
xticklabels({'','Jan','Feb','Mar','Apr','May','Jun','Jul',...
            'Aug','Sep','Oct','Nov','Dec',''})
xlabel('Month')
ylabel('Monthly mean DMS concentration (nM)')

exportfig(gcf,'./FIGs/monthly_mean_NH','fontsize',12,'fontmode',...
          'fixed','color','rgb','renderer','painters')


P1 = plot([1:12],Smean_W20,'ro-','MarkerSize',8,'LineWidth',2);
hold on
P2 = plot(Smean_lana,'g:*','MarkerSize',8,'LineWidth',2);
hold on
P3 = plot(Smean_VS07,'b:+','MarkerSize',8,'LineWidth',2);
hold on
P4 = plot(Smean_SD02,'c:.','MarkerSize',8,'LineWidth',2);
hold on
P5 = plot(Smean_GSM_KD,'k:x','MarkerSize',8,'LineWidth',2);
hold on
P6 = plot(Smean_CHL_KD,'y:x','MarkerSize',8,'LineWidth',2);
hold on
P7 = plot(Smean_GSM_ZLEE,'k:^','MarkerSize',8,'LineWidth',2);
hold on
P8 = plot(Smean_CHL_ZLEE,'y:^','MarkerSize',8,'LineWidth',2);
legend([P1,P2,P3,P4,P5,P6,P7,P8],'This study','L11','VS07','SD02','GSM-KD','CHL-KD','GSM-ZLEE','CHL-ZLEE')
hold off
xlim([0 13])
ylim([0.5 5.5])
xticks([0:13])
xticklabels({'','Jan','Feb','Mar','Apr','May','Jun','Jul',...
            'Aug','Sep','Oct','Nov','Dec',''})
xlabel('Month')
ylabel('Monthly mean DMS concentration (nM)')

exportfig(gcf,'./FIGs/monthly_mean_SH','fontsize',12,'fontmode',...
          'fixed','color','rgb','renderer','painters')

