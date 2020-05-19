% make tracer-tracer plots.
addpath('~/myfun')
load ./output_data/DMS_mod_obs.mat
% first column is index
DMS = squeeze(DMS);
obs = log(DMS(:,1));
mod = log(mean(DMS(:,2:end),2));
% plot M. vs O.
cr = 5:5:95;
W = ones(length(obs),1);
data = [obs,mod];

[bandwidth,density,X,Y] = ...
    mykde2d(data,1000,[-2 -2],[5 5],W);

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
plot([-2 5],[-2 5],'r--','linewidth',2);

subplot('position',[0.82 0.2 0.05 0.6]);
contourf([1 2],cr,[cr(:),cr(:)],cr); hold on
contour([1 2],cr,[cr(:),cr(:)],cr);
hold off
%set(gca,'FontSize',14);
set(gca,'XTickLabel',[]);
set(gca,'YAxisLocation','right');
set(gca,'TickLength',[0 0])
ylabel('(percentile)')

exportfig(gcf,'TF_MvsO','fontsize',12,'fontmode','fixed','color','rgb','renderer','painters')