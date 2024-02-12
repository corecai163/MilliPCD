%% show result
close all
%load('initial_result.mat')

% o1 = incomplete;
% p1 = pred;
% g1 = gtcomplete;
% oc1 = uint8(incolor*255);
% pc1 = uint8(predcolor*255);
% gc1 = uint8(gtcolor*255);
% g1 = squeeze(g1);
% o1 = squeeze(o1);
% p1 = squeeze(p1);
% 
% % gt_pcd = pointCloud(g1);
% % or_pcd = pointCloud(o1);
% % pt_pcd = pointCloud(p1);
% gt_pcd = pointCloud(g1,'Color',gc1);
% or_pcd = pointCloud(o1,'Color',oc1);
% pt_pcd = pointCloud(p1,'Color',pc1);
% %init_pcd = pointCloud(ini_pt);
% 
% 
% figure
% pcshow(pt_pcd,'MarkerSize',53)
% set(gcf,'color','w');
% set(gca,'color','w');
% 
% figure
% pcshow(or_pcd,'MarkerSize',53)
% set(gcf,'color','w');
% set(gca,'color','w');
% 
% figure
% pcshow(gt_pcd,'MarkerSize',53)
% set(gcf,'color','w');
% set(gca,'color','w');
in_pcd = pointCloud(input);
pd_pcd = pointCloud(pred_pcd);
gt_pcd = pointCloud(gt_pcd);

figure
pcshow(in_pcd,'MarkerSize',90)
set(gcf,'color','w');
set(gca,'color','w');

figure
pcshow(pd_pcd,'MarkerSize',90)
set(gcf,'color','w');
set(gca,'color','w');

figure
pcshow(gt_pcd,'MarkerSize',90)
set(gcf,'color','w');
set(gca,'color','w');

