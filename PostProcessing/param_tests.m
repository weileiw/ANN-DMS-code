% in the test we increasing add predictors to the model.
% the purpose of the test is to find out which combination of
% parameters is the best to reporduce DMS concentrations. We keep
% monitoring RMSE for both training and validating data.
clc; clear all; close all;
addpath('~/myfun')
MAE_train = [0.676189; 0.509402; 0.524563; 0.492862; 0.474161; ...
             0.454803; 0.446322; 0.437850; 0.437410; 0.457933; ...
             0.462586];

MAE_test  = [0.643718; 0.494756; 0.526532; 0.471693; 0.469218; ...
             0.475076; 0.455545; 0.475366; 0.461904; 0.488614; ...
             0.491025];

RMSE_train = [0.913077; 0.710745; 0.72278; 0.693568; 0.671465; ...
              0.645273; 0.633595; 0.624832; 0.623125; 0.64732; ...
              0.652418];
              
RMSE_test  = [0.849590; 0.655187; 0.684520; 0.640831; 0.622151; ...
              0.625599; 0.607084; 0.650263; 0.616875; 0.660691; ...
              0.643232];

figure(1)
subplot(2,1,1)
plot(RMSE_train,'ro','MarkerSize',12,'MarkerFaceColor','red')
hold on
plot(RMSE_test ,'b>','MarkerSize',12,'MarkerFaceColor','blue')
grid on
hold off
xlabel('Events')
ylabel('Mean Absolute Error')
subplot(2,1,2)
plot(MAE_train,'ro','MarkerSize',12,'MarkerFaceColor','red')
hold on
plot(MAE_test ,'b>','MarkerSize',12,'MarkerFaceColor','blue')
grid on
xlabel('Events')
ylabel('Root Mean Squear Error')
hold off
exportfig(gcf,'Para_test','color','rgb','fontsize',12,'fontmode','fixed','renderer','painters')

% pbaspect([2 1 1])

