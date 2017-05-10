
% load train-test split
load([rootdir '/data_split/train_' run_id '.txt']);
eval(['train_list=train_' run_id ';']);
load([rootdir '/data_split/test_' run_id '.txt']);
eval(['test_list=test_' run_id ';']);

%% use pretrained model to compute per-layer predictions

savefile=[savedir '/model_tree_e4_run' run_id '.mat'];

  disp('training model on segmentation tree...');
  addpath E:\R2014a\bin\scene_labeling_cvpr2012\code\liblinear-weights-1.8-dense-float\matlab\
  list=train_list;
  collect_tree_data;
  if exist('use_nyu_depth')
    nclass=max(L(:))+1;
    n=accumarray( (L+1), W, [nclass 1] );
    W=W./n(L+1)*10000;
  end
  model_tree=train( W, sparse(L), sparse(F), ['-s 2 -e 0.0001 -c 1 -q'] );
  save(savefile,'model_tree');


% evaluate on first-layer
list=test_list;
collect_tree_data;

k_ucm=k_ucm_all(1);
k_ucm_id=num2str(round(k_ucm*100),'%02d');
segdir=[rootdir '/segmentation/ucm' k_ucm_id];

model=model_tree;

nclass=max(L)+1;
np=length(L);
ps=zeros(np,nclass);
for c=1:nclass,
  ind=find(model.Label==(c-1));
  if  isempty(ind)
      ind = 1;
  end
   for ii=1:1:np
   sum_2=sum(F(ii,:));
   ps(ii,c)=sum_2*model.w(ind,:)';
   end
end

ps(:,1)=-10000;
[dummy,pred]=max(ps,[],2);

n=accumarray( [L+1 pred], W, [nclass nclass] );
c=n./repmat(sum(n,2),1,nclass);

n_region=n;
c_region=c;

