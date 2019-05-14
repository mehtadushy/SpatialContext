function [] = CompileNeighborhoodPoolFeature3DSoftAssignHistogram( imageFileList, dataBaseDir, keypoint_textonSuffix,neighborhood_textonSuffix, histogram_size, params, canSkip, pfig )
% Compile 2D histogram of keypoint-sift labels vs neighbourhood pool histogram
% labels
%   
% 
% imageFileList: cell of file paths
% dataBaseDir: the base directory for the data files that are generated
%  by the algorithm. If this dir is the same as imageBaseDir the files
%  will be generated in the same location as the image file
% featureSuffix: this is the suffix appended to the image file name to
%  denote the data file that contains the labels for the corresponding
%  feature clusters
% dictionarySize: size of descriptor dictionary (
% canSkip: if true the calculation will be skipped if the appropriate data 
%  file is found in dataBaseDir. This is very useful if you just want to
%  update some of the data or if you've added new images.

fprintf('Building 2D Soft Assign Histograms\n\n');

%% parameters

if(~exist('params','var'))
    params.dictionarySize = 200;
end
if(~isfield(params,'dictionarySize'))
    params.dictionarySize = 200;
end

if(~exist('canSkip','var'))
    canSkip = 1;
end


H_all = [];
if(exist('pfig','var'))
    tic;
end
for f = 1:length(imageFileList)

    imageFName = imageFileList{f};
    [dirN, base] = fileparts(imageFName);
    baseFName = fullfile(dirN, base);
    inFName1 = fullfile(dataBaseDir, sprintf('%s%s.mat', baseFName, keypoint_textonSuffix));    
    inFName2 = fullfile(dataBaseDir, sprintf('%s%s.mat', baseFName, neighborhood_textonSuffix));    

    if(mod(f,100)==0 && exist('pfig','var'))
        sp_progress_bar(pfig,3,4,f,length(imageFileList),'Building Histograms:');
    end
    outFName = fullfile(dataBaseDir, sprintf('%s_2dHist_%s_with%s.mat', baseFName, keypoint_textonSuffix, neighborhood_textonSuffix));

    if(exist(outFName,'file')~=0 && canSkip)
        %fprintf('Skipping %s\n', imageFName);
        if(nargout>1)
            load(outFName, 'H');
            H_all = [H_all; H(:)'];
        end
        continue;
    end
    
    %% load texton labels 

    kp_data = load(inFName1, 'texton_ind');
    neigh = load(inFName2, 'texton_ind');
    H = zeros(histogram_size);
    ndata = length(kp_data.texton_ind.x);
    for idx = 1:ndata
        for kp_i = 1:3
            for neigh_i = 1:3
                H(kp_data.texton_ind.data(idx,kp_i), neigh.texton_ind.data(idx,neigh_i))  = H(kp_data.texton_ind.data(idx,kp_i), neigh.texton_ind.data(idx,neigh_i)) + kp_data.texton_ind.weights(idx,kp_i)* neigh.texton_ind.weights(idx,neigh_i) ;
            end
        end
    end
    H = H/ ndata;
    H_all = [H_all; H(:)'];

    %% save texton indices and histograms
    sp_make_dir(outFName);
    save(outFName, 'H');  
end

%% save histograms of all images in this directory in a single file
outFName = fullfile(dataBaseDir, sprintf('2Dhistograms_%s_with%s.mat', keypoint_textonSuffix, neighborhood_textonSuffix));
save(outFName, 'H_all', '-ascii');


end