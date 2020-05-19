clc; clear all; close all
addpath('/Volumes/WANG/DATA');
addpath('./output_data')
load monthly_WSPD_180x360.mat
load surface_tempobs_180x360x12.mat
load  CESM_ICE_fraction_180x360x12
fICE_free = 1-fICE;

M_H2O = 18;    % molecular weight of H2O;
M_S   = 32.13; % sulfur molecular weight;
M_DMS = 62.13; % molecular weight of DMS;
days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31];

Y = [-89.5:1:89.5]; X = [1:1:360];
[Xq,Yq] = meshgrid(X,Y);
% Length of 1 degree of Longitude = ...
% cosine (latitude in decimal degrees) * ...
% length of degree (miles) at equator.

XT = cos(degtorad(Yq)) * 111321;

% Each degree of latitude is about 111 kilometers apart
YT = 111000*ones(180,360);

Area = XT .* YT;

% read all model results;
Files=dir('./output_data/*.mat');
for k=1:length(Files)
    FileNames=Files(k).name;
    DMS_tmp = load(FileNames);
    DMS(:,:,:,k) = DMS_tmp.DMS;
end

% DMS 180x360x12x10
Flux = nan(180,360,12,10);
inty_flux = nan(180,360,12,10);
for kk = 1:10
    for ji = 1:12
        % ICE mask
        ICE2D = fICE_free(:,:,ji);
        % temperature climatology;
        t = T(:,:,ji);
        % wind speed climatology;
        U = WSP(:,:,ji);

        % correction for wind speed.
        eta = sqrt(4*U.^2/pi); % according to Livingstone and Imboden 1993
        xi = 2; % for Rayleigh distribution

        % according to Simo and Dach 2002 GBC;
        U10 = eta .* gamma(1 + 1/xi); % which is equal to mean wind speed;

        % according to GM12;
        Kw660 = 2.1 * U10 - 2.8;
        % Sc number;
        ScDMS = 2674.0 - 147.12 * t + 3.726 * t .^ 2 - 0.038 * t .^ 3 ;
        KwDMS = Kw660 .* (ScDMS ./ 660) .^ (-0.5);
        KwDMS(KwDMS < 0) = 0;
        DMS2D = squeeze(DMS(:,:,ji,kk));

        % calculate flux and
        % convert unit from (umol/m3)*(cm/hr) to umol/m2/day
        Flux_GM12(:,:,ji,kk) = ICE2D .* KwDMS .* DMS2D / 100 * 24;
        inty_tmp = squeeze(Flux_GM12(:,:,ji,kk)) .* Area;
        flux(:,:,ji,kk) = inty_tmp;
        % flux for the N. hemisphere;
        inty_north = inty_tmp(91:end, :);
        sum_GM12_north(ji,kk) = nansum(inty_north(:))*1e-6*M_S* ...
            days_in_month(ji)/1e12;
        % flux for the S. hemisphere;
        inty_south = inty_tmp(1:90, :);
        sum_GM12_south(ji,kk) = nansum(inty_south(:))*1e-6*M_S* ...
            days_in_month(ji)/1e12;
        % flux for the global ocean;
        sum_GM12(ji,kk) = nansum(inty_tmp(:))*1e-6*M_S* ...
            days_in_month(ji)/1e12;
    end
    
    % get flux from another parameterization;
    for ij = 1:12
        % ICE mask
        ICE2D = fICE_free(:,:,ij);
        % temperature climatology;
        t = T(:,:,ij);
        % wind speed climatology;
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
        ka = 659 * U10 .* (M_DMS / M_H2O) ^ (-0.5);
        Gamma = 1 ./ (1 + ka ./ (alpha .* kwDMS));
        % total gas transfer coefficient.
        KwDMS = kwDMS .* (1 - Gamma);
        
        KwDMS(KwDMS < 0) = 0;
        DMS2D = squeeze(DMS(:,:,ij,kk));
        Flux_N00(:,:,ij,kk) = ICE2D .* KwDMS .* DMS2D / 100 * 24;
        % flux(:,:,ij) = Kw660.*DMS2D/100*24;
        inty_tmp = squeeze(Flux_N00(:,:,ij,kk) .* Area);
        % flux for the N. hemisphere;
        inty_north = inty_tmp(91:end, :);
        sum_N00_north(ij,kk) = nansum(inty_north(:))*1e-6*M_S* ...
            days_in_month(ij)/1e12;
        % flux for the S. hemisphere;
        inty_south = inty_tmp(1:90, :);
        sum_N00_south(ij,kk) = nansum(inty_south(:))*1e-6*M_S* ...
            days_in_month(ij)/1e12;
        % flux for the global ocean
        sum_N00(ij,kk) = nansum(inty_tmp(:))*1e-6*M_S* ...
            days_in_month(ij)/1e12;
    end
end

% error prapogation to sum of flux;
mean_GM12 = mean(sum_GM12,2)';
std_GM12 = std(sum_GM12');
mean_N00 = mean(sum_N00,2)';
std_N00  = std(sum_N00');
total_GM12 = sum(mean_GM12);
total_N00  = sum(mean_N00);
error_GM12 = sqrt(sum(std_GM12.^2));
error_N00  = sqrt(sum(std_N00.^2));

% get monthly flux;
Monthly_GM12 = nan(180,360,12);
for ji = 1:12
    % units umol/m2/day
    tmp1 = mean(squeeze(Flux_GM12(:,:,ji,:)),3);
    Monthly_GM12(:,:,ji) = tmp1;
    %
    tmp2 = mean(squeeze(Flux_N00(:,:,ji,:)),3);
    Monthly_N00(:,:,ji) = tmp2;
end

% calculate zonal mean with errors
inter_lat = [1:10:180];
for kk = 1:10
    for ii = 1:12
       % units in Tg/year;
       tmp1(:,:,ii) = squeeze(Flux_GM12(:,:,ii,kk)).*Area*1e-6*M_S*days_in_month(ii)/1e12;
       tmp2(:,:,ii) = squeeze(Flux_N00(:,:,ii,kk)).*Area*1e-6*M_S*days_in_month(ii)/1e12;
    end

    tmp3(:,:,kk) = nansum(tmp1,3);
    tmp4(:,:,kk) = nansum(tmp2,3);

    for jj = 1:length(inter_lat)
        idx = inter_lat(jj);
        tmp3_2D = squeeze(tmp3(:,:,kk));
        tmp5 = tmp3_2D(idx:idx+9,:);
        zonal10_GM12(jj,kk) = nansum(tmp5(:));
        
        tmp4_2D = squeeze(tmp4(:,:,kk));
        tmp6 = tmp4_2D(idx:idx+9,:);
        zonal10_N00(jj,kk) = nansum(tmp6(:));
    end
end
mean_zonal_GM12 = mean(zonal10_GM12,2)';
mean_zonal_N00 = mean(zonal10_N00,2)';
std_zonal_GM12 = std(zonal10_GM12');
std_zonal_N00 = std(zonal10_N00');

% save Flux_CESM_error Monthly_N00 Monthly_GM12 
keyboard
%%%%%%%%%%%%%%%%%%%%%%%%  make figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)

month = linspace(1,12,12);
scatter(month,mean_GM12);
hold on;
errorbar(month, mean_GM12, std_GM12, 'r>','markersize',12,'markerfacecolor','red');
hold on;
scatter(month,mean_N00);
hold on;
errorbar(month, mean_N00, std_N00, 'ko','markersize',12,'markerfacecolor','black');
hold off

xlim([1,12])
xticks([1:1:12])
xlabel('Month')
ylabel('DMS flux (Tg)')

xticklabels({'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug', ...
             'Sep','Oct','Nov','Dec'})

exportfig(gcf,'monthly_DMS_flux_error','fontsize',12,'fontmode', ...
          'fixed','renderer','painters','color','rgb')
%%%%%%%%%%%%%%%%%%%%%%%%  make figures
%%%%%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


