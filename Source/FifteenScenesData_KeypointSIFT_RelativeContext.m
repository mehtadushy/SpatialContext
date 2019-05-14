%%Run the file cell wise
clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_2D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

%Compute keypoint sift descriptors using vl_feat, and normalize the
%descriptors.
% Uncomment if Keypoint Descriptors have not been computed yet
%pfig = sp_progress_bar(sprintf('Computing Keypoint SIFT'));
%GenerateKeypointSiftDescriptors( data.filenames,image_dir,data_dir,[],0,pfig);
%close(pfig);

%Compute relative context of keypoint
params.sigmaFactors = [%45;
                       55;
                       %65;
                       80;
                       ];
params.keypointFeatureSuffix = '_keypoint_sift.mat';
ComputeRelativeKeypointContext(data.filenames, data_dir, params, 0);

%%
%Create different combinations of SIFT and its context, with varying
%weights of the different contributions
canskip = 0;
params.weights = [ %SIFT   %Spatial Context  %Orientation Context
                              1          5               0;
                              1          0               5;
                              1          5               5;
                              1          4               8;
                              1          1               0;
                              1          0               1;
                              1          1               1;];
neighbourhoodSize = [%45;
                       55;
                       %65;
                       80;
                       ];
for i = 1:length(neighbourhoodSize)
        CompileKeypointSIFTContextFeatures(data.filenames, data_dir, sprintf('_keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i)), params, canskip);
end

%%
%Pick 1000 images to do k_means on
params.numTextonImages = 1000;
canskip = 0;
% Cluster the features, assign labels and create histograms of the features
params.weights = [ %SIFT   %Spatial Context  %Orientation Context
                              1          5               0;
                              1          0               5;
                              1          5               5;
                              1          4               8;
                              1          1               0;
                              1          0               1;
                              1          1               1;];
neighbourhoodSize =   [%45;
                       55;
                       %65;
                       80;
                       ];
%dictionarySize = [50, 100, 200, 400];
dictionarySize = [800, 2000, 8000, 15000];

for k = 1:length(dictionarySize)
   params.dictionarySize = dictionarySize(k);
   for i = 1:length(neighbourhoodSize)
    for w = 1:size(params.weights,1)
        sift_weight = params.weights(w,1);
        spatial_weight = params.weights(w,2);
        orientation_weight = params.weights(w,3);
        if (sift_weight > 0)
            sift_string = sprintf('_siftWght_%1.1f',sift_weight);
        else
            sift_string = '';
        end
        if (spatial_weight > 0)
            spatial_string = sprintf('_spatialWht_%1.1f',spatial_weight);
        else
            spatial_string = '';
        end
        if (orientation_weight > 0)
            orientation_string = sprintf('_orientationWht_%1.1f',orientation_weight);
        else
            orientation_string = '';
        end
        
        pfig = sp_progress_bar(sprintf('Context Feat Dic Size %d Neigh Size %d',dictionarySize(k), neighbourhoodSize(i)));
        
        featureSuffix = sprintf('%s%s%s%s', sift_string, spatial_string, orientation_string, sprintf('_keypoint_sift_relative_context_neigh%d.mat', neighbourhoodSize(i)) );
        CalculateDictionary(data.filenames,image_dir,data_dir,featureSuffix,params, canskip, pfig);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,featureSuffix,params, canskip, pfig);
        close(pfig)
    end
   end
end

%%

