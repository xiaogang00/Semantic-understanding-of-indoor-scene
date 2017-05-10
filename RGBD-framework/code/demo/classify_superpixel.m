
%k_ucm=0.08;
k_ucm=round(k_ucm*100)/100;
k_ucm_id=num2str(k_ucm*100,'%02d');


addpath E:\R2014a\bin\scene_labeling_cvpr2012\code\liblinear-weights-1.8-dense-float\matlab

% load train-test split
load([rootdir '\data_split\train_' run_id '.txt']);
eval(['train_list=train_' run_id ';']);
load([rootdir '\data_split\test_' run_id '.txt']);
eval(['test_list=test_' run_id ';']);

% collect training data
disp('collecting training feature matrix...');
train_list = train_list(1:200,:);
test_list = test_list(1:200,:);
F=horzcat(F_all{train_list})';
L=horzcat(L_all{train_list})';
W=horzcat(W_all{train_list})';

% transform features
if exist('use_stanford')
  Fmax=max(F(:,end-10:end),[],1);
  F(:,end-10:end)=F(:,end-10:end)./repmat(Fmax,size(F,1),1);
end
if exist('use_nyu_depth') 
  F(:,1:end-15)=power(abs(F(:,1:end-15)),0.3).*sign(F(:,1:end-15));
  Fmax=max(F(:,end-14:end),[],1);
  F(:,end-14:end)=F(:,end-14:end)./repmat(Fmax,size(F,1),1);
end

savefile=[savedir '\model_kucm' k_ucm_id '_ScaleEx_e4_run' run_id '.mat'];


disp('running liblinear classification... (could take another while)');

if exist('use_stanford')
  model=train( W, sparse(L), sparse(double(F)), '-s 2 -e 0.0001 -c 1 -q' );
end
if exist('use_nyu_depth')
  nclass=max(L(:))+1;
  n=accumarray( (L+1), W, [nclass 1] );
  W=W./n(L+1)*10000;
  model=train( W, sparse(L), sparse(double(F)), '-s 2 -e 0.0001 -c 1 -q' );
end

save(savefile,'model','Fmax');




% run on test images
F=horzcat(F_all{test_list})';
L=horzcat(L_all{test_list})';
W=horzcat(W_all{test_list})';

if exist('use_stanford')
  F(:,end-10:end)=F(:,end-10:end)./repmat(Fmax,size(F,1),1);
end
if exist('use_nyu_depth')
  F(:,1:end-15)=power(abs(F(:,1:end-15)),0.3).*sign(F(:,1:end-15));
  F(:,end-14:end)=F(:,end-14:end)./repmat(Fmax,size(F,1),1);
end

nclass=max(L)+1;
np=length(L);
ps=zeros(np,nclass);
for c=1:nclass,
  ind = find( model.Label == (c-1) );
  for ii=1:1:np
  sum_1=sum(F(ii,:));
  ps(ii,c)=sum_1*model.w(ind,:)';
  end
end

ps(:,1)=-10000;
[dummy,pred]=max(ps,[],2);

n=accumarray( [L+1 pred], W, [nclass nclass] );
c=n./repmat(sum(n,2),1,nclass);

n_region=n;
c_region=c;


