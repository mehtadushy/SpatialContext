% Computes a stronger(than just relative orientation) context that captures
% both relative orientation and types of SIFTs occurring in the
% neighbourhood of a keypoint through a 2D histogram. The relative
% orientation is pushed into the keypoint orientation, while a coarse
% sift histogram from the neighbourhood is clustered orthogonally as 

clear;
image_dir = '~/DirectionalPooling/directionalpooling/Data/SceneCategoriesDataset_Images';
data_dir = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data/SoftAssign_3D';
data_dir2 = '/scratch/common/pool0/dmetha/SceneCategoriesDataset_Data';
load(fullfile(data_dir2, 'scenes_names.mat'));

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

%Compute relative orientation context of keypoint and push back into the keypoint orientation
params.sigmaFactors = [40;
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
dictionarySize = [%20, 40, 50, 100, 
                 200, 400, 800, 1000];
sigmaFactors = [40;
		80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
    parfor k = 1:length(dictionarySize)
        params = struct('numTextonImages', 1000, 'dictionarySize', dictionarySize(k));
        CalculateDictionary(data.filenames,image_dir,data_dir,keypointSuffix, params, canSkip_dictionaryBuild);
        BuildSoftAssignHistograms(data.filenames,image_dir,data_dir,keypointSuffix,params, canSkip_dictionaryBuild);
    end
end
%%
clear params;
% Create coarse SIFT histograms in keypoint neighbourhoods for use as context
canSkip_ContextComputation = 0;
coarseDictionarySize = 400; %20; %[20, 40, 50]; %Coarse Dictionary Size
sigmaFactors = [40;
		80];
for sf = 1:length(sigmaFactors)
   keypointSuffix = sprintf('_%d_neigh_ROcontext_keypoint_sift.mat', sigmaFactors(sf));
    parfor k = 1:length(coarseDictionarySize)
   	textonSuffix = sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift.mat', coarseDictionarySize(k), sigmaFactors(sf));
        params = struct('dictionarySize', coarseDictionarySize(k),'keypointFeatureSuffix',keypointSuffix, 'sigmaFactor', sigmaFactors(sf), 'scaleWeighted', true);
        ComputeCoarseNeighbourhoodHistogram(data.filenames,data_dir,textonSuffix, params, canSkip_ContextComputation);
    end
end
clear params;
%
% Cluster Keypoint Context
%Pick 1000 images to do k_means on
dictionarySize = [20 , 80, 200]; %[10, 20 ,40]; % Context Dictionary Size
%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
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


% Next up, we would construct 2D Histograms 
neighbourhoodSize = [ 40;
                       80;
                       ];
%coarseDictionarySize = [20, 40, 50]; %Coarse Dictionary Size
contextDictionarySize = [20, 80, 200];%[10 ,20, 40];
keypointDictionarySize = [%50, 100, 
			%200, 400,
			800, 1000];
parfor j = 1: length(keypointDictionarySize)
    for l = 1: length(coarseDictionarySize)
     for k = 1: length(contextDictionarySize)
      for i = 1:length(neighbourhoodSize)
            pfig = sp_progress_bar(sprintf('2D Hist of Neigh Size%d Keypt Dic%d Context Dic%d',neighbourhoodSize(i), keypointDictionarySize(j), contextDictionarySize(k)));
            CompileNeighborhoodPoolFeature2DSoftAssignHistogram( data.filenames,data_dir, sprintf('_soft_texton_ind_%d_%d_neigh_ROcontext_keypoint_sift',keypointDictionarySize(j),neighbourhoodSize(i)), sprintf('_soft_texton_ind_%d_%d_neigh_contexthist%d', contextDictionarySize(k), neighbourhoodSize(i), coarseDictionarySize(l)), [keypointDictionarySize(j), contextDictionarySize(k)],[] , 0, pfig );
            close(pfig)
      end
    end
    end
end

%
predict_test_2D_stronger_context


