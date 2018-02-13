% clear
% num = 0;
% load annotation_list.mat
dst_dir = '/home/jihang/Jiahong/pix2pix-tensorflow/OCT_data/';
ori_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';
ann_dir = '/home/jihang/Jiahong/data/OCT/Annotations/';

margin = 0;
low = 30;
high = 80;
block = 256

for j = 5 : 5
    ori_path = [ori_dir num2str(j) '/original/']
    ann_path = [ann_dir num2str(j) '/originalAnnotationTejas_new/'];
    ann_directory = dir(ori_path);
    for i = 1 : length(ann_directory)
%     for i = 1 : 10 : length(ann_directory)
        if(isequal(ann_directory(i).name, '.') || isequal(ann_directory(i).name, '..'))   %exclude hidden folder & not folder
            continue;
        end
        num = num + 1;
        img_ori_path = [ori_path ann_directory(i).name]; %all images path
        img_ori = imread(img_ori_path);
%         img_ann_path = [ann_path ann_directory(i).name];
%         img_ann = imread(img_ann_path);
        img_ann = img_ori;
        [row col cha] = size(img_ann);
%         figure(1)
%         imshow(img_ann);
        img_clear = img_ori;
	img_mask = img_ori;

            for k = 1 : floor(col/block)
                img_ori_tmp = img_ori(:, (k-1)*block+1:k*block);
                img_clear_tmp = img_clear(:, (k-1)*block+1:k*block);
		img_mask_tmp = img_clear_tmp;
                img_dst_path = [dst_dir 'test_testdata_clean/' num2str(num, '%04d') '_' num2str(k) '.png'];
                img = [img_ori_tmp img_clear_tmp img_mask_tmp];
%                 img = repmat(img,1,1,3);
                imwrite(img, img_dst_path);
            end
            if(col ~= 100 * k)
                img_ori_tmp = img_ori(:, col-block+1:col);
                img_clear_tmp = img_clear(:, col-block+1:col);
		img_mask_tmp = img_clear_tmp;
                img_dst_path = [dst_dir 'test_testdata_clean/' num2str(num, '%04d') '_' num2str(k+1) '.png'];
                img = [img_ori_tmp img_clear_tmp img_mask_tmp];
%                 img = repmat(img,1,1,3);
                imwrite(img, img_dst_path);
            end
%         end
    end
end
