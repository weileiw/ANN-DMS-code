clc; clear all; close all
addpath('~/Dropbox/myfunc');
addpath('/Volumes/WANG/DATA');
load monthly_WSPD_180x360.mat
load surface_tempobs_180x360x12.mat
load  CESM_ICE_fraction_180x360x12
load ../Lana_Gali_VS_SD/Gali_climat.mat CHL_KD CHL_ZLEE GSM_KD ...
    GSM_ZLEE SD02 VS07
% load ../Lana_Gali_VS_SD/DMS_climate_lana_180x360.mat DMS
% DMS = CHL_KD;
% DMS = CHL_ZLEE;
% DMS = GSM_KD;
% DMS = GSM_ZLEE;
DMS = SD02;
% DMS = VS07;

fICE_free = 1 - fICE;

M_H2O = 18;    % molecular weight of H2O;
M_DMS = 62.13; % molecular weight of DMS;
M_S   = 32.13; % sulfur molecular weight;
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31];

Y = [-89.5:1:89.5]; X = [1:1:360];
[Xq,Yq] = meshgrid(X,Y);
% Length of 1 degree of Longitude = ...
% cosine (latitude in decimal degrees) * ...
% length of degree (miles) at equator.

XT = cos(degtorad(Yq)) * 111321;

% Each degree of latitude is about 111 kilometers apart
YT = 111000 * ones(180, 360);

Area = XT.*YT;
seed = [2,4,8,16,32,64,128,256,512,1024];

Flux_GM12 = nan(180, 360, 12);
inty_flux = nan(180, 360, 12);

% get flux from GM12;
for kk = 1:12
    % ICE mask
    ICE2D = fICE_free(:, :, kk);
    % temperature
    t = T(:, :, kk);
    % GM12 parameterization;
    Kw660 = 2.1 * WSP(:,:,kk) - 2.8;
    % Sc number
    ScDMS = 2674.0 - 147.12*t + 3.726*t.^2 - 0.038*t.^3 ;
    KwDMS = Kw660 .* (ScDMS ./ 660) .^ (-0.5);
    KwDMS(KwDMS < 0) = 0;
    DMS2D = DMS(:, :, kk);
    
    % calculate flux and 
    % convert unit from (umol/m3)*(cm/hr) to umol/m2/day
    flux(:,:,kk) = ICE2D .* KwDMS .* DMS2D / 100 * 24;
    inty_tmp = flux(:, :, kk) .* Area;
    Flux_GM12(:, :, kk) = flux(:, :, kk); % umol/m2/day
    sum_GM12(kk) = nansum(inty_tmp(:)) * 1e-6 * M_S * ...
        days_in_month(kk) / 1e12;
end

% get flux from another parameterization;
for ij = 1:12
    % ICE mask
    ICE2D = fICE_free(:,:,ij);
    % temperature climatology
    t = T(:,:,ij);
    % wind speed climatology
    U = WSP(:,:,ij);
    
    % correction for wind speed.
    eta = sqrt(4*U.^2/pi); % according to Livingstone and Imboden 1993
    xi = 2; % for Rayleigh distribution
    U_squared = eta.^2 .* gamma(1 + 2/xi); % according to Simo and
                                           % Dach 2002;
    U10 = eta .* gamma(1 + 1/xi); % which is equal to mean wind speed;
    k600 = 0.222*U_squared + 0.333*U10; 

    ScDMS = 2674.0 - 147.12*t + 3.726*t.^2 - 0.038*t.^3 ;
    kwDMS = k600 .* (ScDMS ./ 600) .^ (-0.5);
    % Ostwald coefficient according to McGillis et al., 2000
    alpha = exp(3525 ./ (t + 273.15) - 9.464);
    % air-side transfer coefficient ka
    ka = 659 * U10 .* (M_DMS ./ M_H2O) .^ (-0.5);
    Gamma = 1 ./ (1 + ka ./ (alpha .* kwDMS));
    % total gas transfer coefficient.
    KwDMS = kwDMS .* (1 - Gamma);
    
    KwDMS(KwDMS<0) = 0;
    DMS2D = DMS(:,:,ij);
    flux(:,:,ij) = ICE2D .* KwDMS .* DMS2D / 100 * 24;
    inty_tmp = flux(:,:,ij) .* Area;
    Flux_N00(:,:,ij) = flux(:,:,ij); % umol/m2/day
    sum_N00(ij) = nansum(inty_tmp(:)) * 1e-6 * M_S * ...
        days_in_month(ij) / 1e12;
end

% error prapogation to sum of flux;
total_GM12 = sum(sum_GM12);
total_N00  = sum(sum_N00);

% get monthly flux;
% units umol/m2/day
Monthly_GM12 = Flux_GM12;
Monthly_N00  = Flux_N00;


% calculate zonal mean with errors
inter_lat = [1:10:180];

for ii = 1:12
    % units in Tg/year;
    tmp1(:,:,ii) = squeeze(Flux_GM12(:,:,ii)).*Area*1e-6*M_S*days_in_month(ii)/1e12;
    tmp2(:,:,ii) = squeeze(Flux_N00(:,:,ii)).*Area*1e-6*M_S*days_in_month(ii)/1e12;
end

for jj = 1:length(inter_lat)
    idx = inter_lat(jj);
    tmp1_2D = nansum(tmp1,3);
    tmp3 = tmp1_2D(idx:idx+9,:);
    zonal10_GM12(jj) = nansum(tmp3(:));

    tmp2_2D = nansum(tmp2,3);
    tmp4 = tmp2_2D(idx:idx+9,:);
    zonal10_N00(jj) = nansum(tmp4(:));
end
% save Gali_zonal4 zonal10_N00
% save Flux_CESM_error Monthly_N00 Monthly_GM12 

%%%%%%%%%%%%%%%%%%%%%%%%  make figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure(1)

% month = linspace(1,12,12);
% plot(sum_GM12,'ro');
% hold on;
% plot(sum_N00,'b*');
% hold off

% xlim([1,12])
% xticks([1:1:12])
% xlabel('Month')
% ylabel('DMS flux (Tg)')

% xticklabels({'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug', ...
             % 'Sep','Oct','Nov','Dec'})

% exportfig(gcf,'monthly_DMS_lana','fontsize',12,'fontmode', ...
          % 'fixed','renderer','painters','color','rgb')
%%%%%%%%%%%%%%%%%%%%%%%%  make figures
%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


