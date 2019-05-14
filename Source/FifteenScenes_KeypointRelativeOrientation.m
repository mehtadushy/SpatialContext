% A different approach to adding relative context to features by modifying
% the reference direction of keypoints by looking at the neighbourhoods,
% and then extracting SIFT features using the modified reference directions.
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_RotImages3';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'rot3_scenes_names.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Flags
DoG_keypoints = false; % uses DoG keypoints for SIFT
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


%Compute relative context of keypoint
params.sigmaFactors = [40;
                       80;
                       ];
params.keypointFrameSuffix = '_keypoint_sift_frame.mat';
params.completeNeighbourhood = true;

ComputeRelativeOrientationCorrectedFrame(data.filenames, data_dir, params, canSkip_FrameComputation);


% Now use the frames to compute Keypoint SIFT
sigmaFactors = params.sigmaFactors;

canSkip_SIFTComputation = 0;

for sf = 1:length(sigmaFactors)
    %clear params;
    params.keypointFramePrefix = sprintf('_%d_neigh_ROcontext', sigmaFactors(sf));
    GenerateKeypointSiftFromFrames(data.filenames,image_dir,data_dir, params, canSkip_SIFTComputation);
end
%RemoveCountFromDirContext(data.filenames, data_dir, params, 0);


%%
% Cluster Keypoint SIFT
%Pick 1000 images to do k_means on
canSkip_dictionaryBuild = 0;

clear params;
dictionarySize = [50, 100, 200, 400, 600, 800, 1000, 1500];%, 2000, 2500];
sigmaFactors = [40; 80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));

    parfor k = 1:length(dictionarySize)
        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
%        params.numTextonImages = 1000;
        %pfig = sp_progress_bar(sprintf('Computing RO Context SIFT Dict With Size %d',dictionarySize(k)));
        %params.dictionarySize = dictionarySize(k);
        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
        %close(pfig)
    end
    
end



%Using LibSVM Train

load(fullfile(data_dir2, 'rot3_category_names.mat'));
%Do a test-train split (data is already randomized
%train_percent = 90;
%train.index = (1 : floor(length(data.labels)*train_percent/100));
%train.labels = data.labels(train.index);
%test.index = (length(train.index)+1 : length(data.labels));
%test.labels = data.labels(test.index);
train_test_datafilename = fullfile(data_dir, 'train_test_data.mat');
save(train_test_datafilename, 'train', 'test');
%%

%Pick 1000 images to do k_means on

neighbourhoodSize = sigmaFactors;


accuracy_results = cell( length(dictionarySize)+1, length(neighbourhoodSize)+1);


for k = 1:length(dictionarySize)
   params.dictionarySize = dictionarySize(k);
   accuracy_results{ k+1, 1} = sprintf('%d Dic',dictionarySize(k));
   for i = 1:length(neighbourhoodSize)
        accuracy_results{1, i+1} = sprintf('%dN',neighbourhoodSize(i));
   end
end




%%
for i = 1:length(neighbourhoodSize)
   acc_log = cell(length(dictionarySize),1);
     parfor k = 1:length(dictionarySize)

       fprintf('Predicting With %d Dictionary %d Neigh Size', dictionarySize(k), neighbourhoodSize(i));

            inFName = fullfile(data_dir, sprintf('soft_histograms_%d_%d_neigh_ROcontext_keypoint_sift.mat', dictionarySize(k),neighbourhoodSize(i) ));
            H_all = load(inFName,'-ascii');
            dat = load(train_test_datafilename);
            train = dat.train;
            test = dat.test;
            train_features = H_all(train.index,:);
            test_features = H_all(test.index,:);

            %We'll do multiclass classification
            %Compute kernel for training data
            train.K = [(1:size(train_features,1))' , intersectionKernel(train_features, train_features)];
            test.K = [(1:size(test_features,1))' , intersectionKernel(test_features, train_features)];
    
            bestcv = 0;
            for log2c = -2:2:4
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; bestc = 2^log2c; bestlog2c = log2c;
                end
            end
            fprintf('Best C = %d\n',bestc);
            bestcv = 0;
            for log2c = (bestlog2c-0.3):0.1:(bestlog2c+0.3)
                params = sprintf('-q -c %d -v 2 -t 4', 2^log2c);
                cv = svmtrain(train.labels',train.K, params );
                if (cv >= bestcv),
                  bestcv = cv; C = 2^log2c;
                end
            end
            fprintf('C = %d\n',bestc);
            params = sprintf('-q -c %d -t 4', C);
            bow_model = svmtrain(train.labels',train.K, params );
            [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test.labels', test.K, bow_model);
            acc_log{k} = accuracy_P(1);

     end
            accuracy_results(2:end ,i+1) = acc_log;
end


outfname = fullfile( data_dir, 'results', 'ROcontext_sift_accuracy.mat');
save(outfname, 'accuracy_results');







