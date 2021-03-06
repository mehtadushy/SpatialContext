% Uses a histogram of "phrase" features, which are histograms of
% keypoint sift occurring in the scaled neighbourhoods of keypoints of
% an image. Prior to computing SIFT descriptors, the relative
% orientation context is pushed into the keypoint orientation. This is
% essentially the 2D Histogram marginalized along the appearance dimension.
% BoW of SIFT Features (slighty denser) is appended to this new "phrase"
% feature, and tf-idf weighting employed to select important features

clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Flags
DoG_keypoints = true; % uses DoG keypoints for SIFT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

canSkip_FrameComputation = 0;
params.axisAlignedSIFT = false;
%Compute keypoint sift frames
pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT Frames'));
if(DoG_keypoints==true)
   GenerateDoGKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
else
   GenerateKeypointSiftFrames( data.filenames,image_dir,data_dir,params,canSkip_FrameComputation,pfig);
end
close(pfig);

%Compute relative orientation context of keypoint and push back into the keypoint orientation
params.sigmaFactors = [%40;
                      80;
                      ];
params.keypointFrameSuffix = '_keypoint_sift_frame.mat';

ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, canSkip_FrameComputation);

% Now use the frames to compute Keypoint SIFT
sigmaFactors = params.sigmaFactors;

canSkip_SIFTComputation = 0;

for sf = 1:length(sigmaFactors)
   %clear params;
   params.keypointFramePrefix = sprintf('_%d_neigh_ROcontext', sigmaFactors(sf));
   GenerateKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
end


%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [400, 800, 1000, 1500]; %[%20, 40, 50, 100, 
                 %200, 400, 800, 1000];
sigmaFactors = [%40;
		80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
    parfor k = 1:length(dictionarySize)
        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
    end
end

clear params;
% Create coarse SIFT histograms in keypoint neighbourhoods for use as context
canSkip_ContextComputation = 0;
coarseDictionarySize = 400;%[200, 400, 800]; %20; %[20, 40, 50]; %Coarse Dictionary Size
sigmaFactors = [40;
		80];
for sf = 1:length(sigmaFactors)
  keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', 80);
   parfor k = 1:length(coarseDictionarySize)
  	textonSuffix = sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift.mat', coarseDictionarySize(k), 80);
       params = struct('dictionarySize', coarseDictionarySize(k),'keypointFeatureSuffix',keypointSuffix, 'sigmaFactor', sigmaFactors(sf), 'scaleWeighted', true);
       ComputeCoarseNeighbourhoodHistogram(data.filenames,data_dir,textonSuffix, params, canSkip_ContextComputation);
   end
end
%%
clear params;

% Cluster Keypoint Context
%Pick 1000 images to do k_means on
dictionarySize = [800, 1000, 1500];%[50 , 100, 200, 400, 800]; %[10, 20 ,40]; % Context Dictionary Size
coarseDictionarySize = [400]; %Coarse Dictionary Size
neighbourhoodSize = [40; 
		     80];
parfor k = 1:length(dictionarySize)
 for j = 1:length(coarseDictionarySize)
    for i = 1:length(neighbourhoodSize)
    pfig =                     sp_progress_bar(sprintf('Computing Keypoint Context Dictionaries With Dic Size %d',dictionarySize(k)));
    para = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
    CalculateHistogramDictionary(data.filenames,data_dir,sprintf('_%d_neigh_contexthist%d.mat', neighbourhoodSize(i), coarseDictionarySize(j)),para, 0, pfig);
    BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,sprintf('_%d_neigh_contexthist%d.mat', neighbourhoodSize(i), coarseDictionarySize(j)),para, 0, pfig);
    close(pfig)
    end
 end
end
%%
load(fullfile(data_dir2, 'scenes_names.mat'));
load(fullfile(data_dir2, 'category_names.mat'));
%Do a test-train split (data is already randomized
train_percent = 90;
train.index = (1 : floor(length(data.labels)*train_percent/100));
train.labels = data.labels(train.index);
test.index = (length(train.index)+1 : length(data.labels));
test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');

%%
%%%%%%%%%%%%%%%%
%  Histograms of context
%%%%%%%%%%%%%%%%
neighbourhoodSize = [ 40;
                       80;
                       ];
contextDictionarySize = [800, 1000, 1500];%[50 , 100, 200, 400, 800];
coarseDictionarySize = 400; %Has to be one of 200, 400, 800
c_sz = length(contextDictionarySize);
k_sz = length(coarseDictionarySize);
n_sz=1;
accuracy_results = cell(length(contextDictionarySize)+1, length(neighbourhoodSize)*k_sz+1);
config_table = zeros(n_sz*k_sz*c_sz, 3);
idx=1;

for k = 1: length(contextDictionarySize)
     accuracy_results{k+1, 1} = sprintf('%d_Context', contextDictionarySize(k));
     for i = 1:n_sz
          for j = 1: length(coarseDictionarySize)
            accuracy_results{1, (i-1)*length(coarseDictionarySize)+j+1} = sprintf('%d_CD', coarseDictionarySize(j));
            config_table(idx,:) = [contextDictionarySize(k),0, coarseDictionarySize(j)];
            idx= idx+1;
          end      
     end
end

%%

classes = unique(train.labels);

acc_log = cell(1, size(config_table, 1));

parfor i = 1: size(config_table, 1)
            
            inFName = fullfile(data_dir, sprintf('soft_histograms_%s.mat',  sprintf('%d_%d_neigh_contexthist%d', config_table(i,1), 80, config_table(i,3))));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);
            inFName = fullfile(data_dir, sprintf('soft_histograms_%s.mat',  sprintf('%d_%d_neigh_contexthist%d', config_table(i,1), 40, config_table(i,3))));
            H_all = load(inFName,'-ascii');
            train_features = [train_features, H_all(train.index,:)];
            test_features = [test_features, H_all(test.index,:)];          
            inFName = fullfile(data_dir, sprintf('soft_histograms_%s.mat',  sprintf('%d_%d_neigh_ROcontext_keypoint_sift', config_table(i,1), 80)));
            H_all = load(inFName,'-ascii');
            train_features = [train_features, H_all(train.index,:)];
            test_features = [test_features, H_all(test.index,:)];
            
%             idf = log(size(train_features,1)./ (sum(train_features>0,1)+1));
%             train_features(train_features~=0) = 1+ log( train_features(train_features~=0) );
%             test_features(test_features~=0) = 1+ log( test_features(test_features~=0) );
%             train_features = bsxfun( @(A,B) A.*B, idf, train_features);
%             test_features = bsxfun( @(A,B) A.*B, idf, test_features);
%             
%             %We'll do multiclass classification
%             %Compute kernel for training data
             train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
             test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
            
            
%             sqrt_features = sqrt(train_features);
%             % Compute feature density ratios of one class vs rest and use
%             % the most distinctive features for classification with svm
%             feat_idx = [];
%             for c = 1: length(classes)
%                 class = classes(c);
%                 feat1 = sqrt_features(train.labels == class,:);
%                 feat2 = sqrt_features(train.labels ~= class,:);
%                 density = mean(feat1)./mean(feat2);
%                 imp = find(density > 1)
%                 feat_idx = [feat_idx find(imp>0)];                
%             end
%    
%             feat_idx = unique(feat_idx);


%             sqrt_features = sqrt(train_features);
%            % Compute feature density ratios of one class vs rest and use
%            % the most distinctive features for classification with svm
%            feat_idx = [];
%            mean_c = zeros(length(classes), size(train_features,2));
% 
%            for c = 1: length(classes)
%                class = classes(c);
%                mean_c(c,:) = mean(sqrt_features(train.labels == class,:));
%            end
%           for c = 1: length(classes)
%                 complement_mean = max(mean_c( ~((1:length(classes)) == c) , :));
%                 density = mean_c(c,:) ./ complement_mean;
%                 imp = find(density > 1);
%                 feat_idx = [feat_idx find(imp>0)];  
%           end
%            feat_idx = unique(feat_idx);

            %We'll do multiclass classification
            %Compute kernel for training data
             %train.K = [(1:size(train_features,1))' , intersectionKernel(train_features(:,feat_idx), train_features(:,feat_idx))];
             %test.K = [(1:size(test_features,1))' , intersectionKernel(test_features(:,feat_idx), train_features(:,feat_idx))];
    
            bow_model = svmtrain(train.labels',train.K, '-t 4' );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{i} = accuracy_P(1);
end

idx=1;
for k = 1: length(contextDictionarySize)
     for i = 1:n_sz
          for j = 1: length(coarseDictionarySize)
            accuracy_results(k+1, (i-1)*length(coarseDictionarySize)+j+1) = acc_log(idx);
            idx= idx+1;
          end      
     end
end


outfname = fullfile( data_dir, 'results', 'AppendedContextHist_accuracy.mat');
save(outfname, 'accuracy_results');

