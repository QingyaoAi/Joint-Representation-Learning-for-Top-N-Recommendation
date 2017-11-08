# Overview #
This is an implementation of the Joint Representation Learning Model (JRL) for product recommendation based on heterogeneous information sources [2]. Please cite the following paper if you plan to use it for your project：
    
* Yongfeng Zhang* Qingyao Ai*, Xu Chen, W. Bruce Croft. 2017. Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources. In Proceedings of CIKM ’17
    	
The JRL is a deep neural network model that jointly learn latent representations for products and users based on reviews, images and product ratings. 
The model can jointly or independently latent representations for products and users based on different information.
The probability (which is also the rank score) of an product being purchased by a user can be computed with their concatenated latent representations from different information sources. 
Please refer to the paper for more details.

### Requirements: ###
    1. To run the JRL model in ./JRL/ and the python scripts in ./scripts/, python 2.7+ and Tensorflow v1.0+ are needed
    2. To run the jar package in ./jar/, JDK 1.7 is needed
    3. To compile the java code in ./java/, galago from lemur project (https://sourceforge.net/p/lemur/wiki/Galago%20Installation/) is needed. 

### Data Preparation ###
    1. Download Amazon review datasets from http://jmcauley.ucsd.edu/data/amazon/ (e.g. In our paper, we used 5-core data).
    2. Stem and remove stop words from the Amazon review datasets if needed (e.g. In our paper, we stem the field of "reviewText" and "summary" without stop words removal)
        1. java -Xmx4g -jar ./jar/AmazonReviewData_preprocess.jar <jsonConfigFile> <review_file> <output_review_file>
            1. <jsonConfigFile>: A json file that specify the file path of stop words list. An example can be found in the root directory. Enter "false" if don’t want to remove stop words. 
            2. <review_file>: the path for the original Amazon review data
            3. <output_review_file>: the output path for processed Amazon review data
    3. Index datasets
        1. python ./scripts/index_and_filter_review_file.py <review_file> <indexed_data_dir> <min_count>
            1. <review_file>: the file path for the Amazon review data
            2. <indexed_data_dir>: output directory for indexed data
            3. <min_count>: the minimum count for terms. If a term appears less then <min_count> times in the data, it will be ignored.
    4. Split train/test
        1. Download the meta data from http://jmcauley.ucsd.edu/data/amazon/ 
        2. Split datasets for training and test
            1. python ./scripts/split_train_test.py <indexed_data_dir> <review_sample_rate>
            2. <indexed_data_dir>: the directory for indexed data.
            3. <review_sample_rate>: the proportion of reviews used in test for each user (e.g. in our paper, we used 0.3).
	5. Match image features
		1. Download the image features from http://jmcauley.ucsd.edu/data/amazon/ .
		2. Match image features with product ids.
			1. python ./scripts/match_with_image_features.py <indexed_data_dir> <image_feature_file>
			2. <indexed_data_dir>: the directory for indexed data.
			3. <image_feature_file>: the file for image features data.
	6. Match rating features
		1. Construct latent representations based on rating information with any method you like (e.g. BPR).
		2. Format the latent factors of items and users in "item_factors.csv" and "user_factors.csv" such that each row represents one latent vector for the corresponding item/user in the <indexed_data_dir>/product.txt.gz and user.txt.gz (see the example csv files).
		3. Put the item_factors.csv and user_factors.csv into <indexed_data_dir>
		

### Model Training/Testing ###
    1. python ./JRL/main.py --<parameter_name> <parameter_value> --<parameter_name> <parameter_value> … 
        1. learning_rate:  The learning rate in training. Default 0.05.
        2. learning_rate_decay_factor: Learning rate decays by this much whenever the loss is higher than three previous loss. Default 0.90
        3. max_gradient_norm: Clip gradients to this norm. Default 5.0
        4. subsampling_rate: The rate to subsampling. Default 1e-4. 
        5. L2_lambda: The lambda for L2 regularization. Default 0.0
        6. image_weight: The weight for image feature based training loss (see the paper for more details).
        7. batch_size: Batch size used in training. Default 64
        8. data_dir: Data directory, which should be the <indexed_data_dir>
        9. input_train_dir: The directory of training and testing data, which usually is <data_dir>/query_split/
        10. train_dir: Model directory & output directory
        11. similarity_func: The function to compute the ranking score for an item with the joint model of query and user embeddings. Default "product".
            1. "product": the dot product of two vectors.
            2. "cosine": the cosine similarity of two vectors.
            3. "bias_product": the dot product plus a item-specific bias
        12. net_struct:  Network structure parameters. Different parameters are separated by "_" (e.g. ). Default "simplified_fs"
            1. "bpr": train models in a bpr framework [1].
            2. "simplified": simplified embedding-based language models without modeling for each review [2].
            3. "hdc": use regularized embedding-based language models with word context [4]. Otherwise, use the default model, which is the embedding-based language models based on paragraph vector model. [3]
            5. "extend": use the extendable model structure (see more details in the paper).
            6. "text": use review data. 
            7. "image": use image data.
            8. "rate": use rating-based latent representations.
            	* if none of "text", "image" and "rate" is specified, the model will use all of them.	
        13. embed_size: Size of each embedding. Default 100.
        14. window_size: Size of context window for hdc model. Default 5.
        15. max_train_epoch: Limit on the epochs of training (0: no limit). Default 5.
        16. steps_per_checkpoint: How many training steps to do per checkpoint. Default 200
        17. seconds_per_checkpoint: How many seconds to wait before storing embeddings. Default 3600
        18. negative_sample: How many samples to generate for negative sampling. Default 5.
        19. decode: Set to "False" for training and "True" for testing. Default "False"
        20. test_mode: Test modes. Default "product_scores"
            1. "product_scores": output ranking results and ranking scores; 
            2. "output_embedding": output embedding representations for users, items and words.
        21. rank_cutoff: Rank cutoff for output rank lists. Default 100.
    2. Evaluation
        1. After training with "--decode False", generate test rank lists with "--decode True".
        2. TREC format rank lists for test data will be stored in <train_dir> with name "test.<similarity_func>.ranklist"
        3. Evaluate test rank lists with ground truth <input_train_dir>/test.qrels.
        	1. python recommendation_metric.py <rank_list_file> <test_qrel_file> <rank_cutoff_list>
        	2. <rank_list_file>: the result list, e.g. <train_dir>/test.<similarity_func>.ranklist 
        	3. <test_qrel_file>: the ground truth, e.g. <input_train_dir>/test.qrels .
        	4. <rank_curoff_list>: the number of top documents to used in evaluation, e.g. NDCG@10 -> rank+cutoff_list=10.

### Reference: ###
    [1] Ste en Rendle, C. Freudenthaler, Zeno Gantner, and Lars Schmidtieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In UAI.
    [2] Yongfeng Zhang* Qingyao Ai*, Xu Chen, W. Bruce Croft. 2017. Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources. In Proceedings of CIKM ’17
    [3] Quoc V Le and Tomas Mikolov. 2014. Distributed Representations of Sentencesand Documents.. In ICML
    [4] Sun, Fei, Jiafeng Guo, Yanyan Lan, Jun Xu, and Xueqi Cheng. 2015 "Learning Word Representations by Jointly Modeling Syntagmatic and Paradigmatic Relations." In ACL 
    [5] Ivan Vulic ́ and Marie-Francine Moens. 2015. Monolingual and cross-lingual in-formation retrieval models based on (bilingual) word embeddings. In Proceedingsof the 38th ACM SIGIR