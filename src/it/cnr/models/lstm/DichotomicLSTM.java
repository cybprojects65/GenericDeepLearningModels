package it.cnr.models.lstm;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.bytedeco.javacpp.Pointer;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.SelfAttentionLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.LocationPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.shade.guava.io.Files;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import it.cnr.models.MultiLayerNetworkWhiteBox;
import it.cnr.models.MultiLayerNetworkWhiteBoxSerializer;
import it.cnr.models.Utils4Models;

public class DichotomicLSTM {
	private static final Logger log = LoggerFactory.getLogger(DichotomicLSTM.class);
	/**
	 * Prepare files for training and test like this: train ----features
	 * -----0.csv,1.csv,..,M.csv ----labels -----0.csv,1.csv,..,M.csv CSV files for
	 * features should be: rowsXcolumns where rows=one feature vector per analysis
	 * window columns=one colum for each feature in the window vector E.g., for a 4
	 * window signal with 3-dimensional vector for each window 1,2,3 2,1,3 3,2,1
	 * 1,1,1
	 * 
	 * NOTE: no header is required CSV files for features should be plain text with
	 * just one number corresponding to the index of the class E.g., for two classes
	 * it should contain either 0 or 1
	 * 
	 * @param args
	 */
	
	public static File testFile = new File("./dichotomicsimulatedseries/test/features/3.csv");
	DataNormalization normalizer = null;
	MultiLayerNetworkWhiteBox net = null;
	
	private File baseDir;
	private File baseTrainDir;
	private File featuresDirTrain;
	private File labelsDirTrain;
	private File baseTestDir;
	private File featuresDirTest;
	private File labelsDirTest;
	public int nfilesTraining;
	public int nfilesTest;
	public int nhidden;
	public int numLabelClasses;
	public int miniBatchSize;
	public int nEpochs;
	public File savedNet;
	public File savedNormalizer;
	public double currentAttentionScore;

	
	public static void main(String[] args) throws Exception{
		DichotomicLSTM lstm = new DichotomicLSTM();
		int nhidden = 3;
		int nClasses = 2;
		int minibatch = 2;
		int nEpochs = 3;
		
		lstm.init(new File("./dichotomicsimulatedseries"),
				nhidden, nClasses, minibatch, nEpochs);
		//lstm.train();
		
		lstm = new DichotomicLSTM();
		lstm.init(new File("./dichotomicsimulatedseries"),
				nhidden, nClasses, minibatch, nEpochs);
		//lstm.solve(testFile);
		lstm.classify(testFile);
	}

	
	public void init(File baseDir, int nhidden, int numLabelClasses, int miniBatchSize, int nEpochs) {
		this.baseDir = baseDir;
		
		this.baseTrainDir = new File(baseDir, "train");
		this.featuresDirTrain = new File(baseTrainDir, "features");
		this.labelsDirTrain = new File(baseTrainDir, "labels");
		this.baseTestDir = new File(baseDir, "test");
		this.featuresDirTest = new File(baseTestDir, "features");
		this.labelsDirTest = new File(baseTestDir, "labels");
		
		new File("./logs/application.log").delete();
		this.nfilesTraining = featuresDirTrain.listFiles().length;
		this.nfilesTest = featuresDirTest.listFiles().length;
		this.nhidden = nhidden;
		this.numLabelClasses = numLabelClasses;
		this.miniBatchSize = miniBatchSize;
		this.nEpochs = nEpochs;

		this.savedNet = new File(baseDir, "net_" + nhidden + "hidden_" + nEpochs + "epo_" + miniBatchSize + "batch_"
				+ numLabelClasses + "classes.bin");
		this.savedNormalizer = new File(baseDir, savedNet.getName().replace(".bin", "_normalizer.bin"));

	}

	public double accuracy = 0;
	public String evaluation = "";
	public void train() throws Exception {
		
		//sequences setup
		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
		trainFeatures.initialize(
				new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, nfilesTraining-1));
		SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
		trainLabels.initialize(
				new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, nfilesTraining-1));
		DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
	            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
		
		//Utils4Models.printDataSet(trainData);
		
		trainData.reset();
		int nInput = trainData.next().numInputs();
		log.info("Input vector length: "+nInput);
		
		//normalize training data
		normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        
        
        
        //Utils4Models.movingAverage(trainData, 2);
        Utils4Models.showData(trainData);
        
        trainData.reset();
        trainData.setPreProcessor(normalizer);
		
        //normalize test data
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, nfilesTest-1));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, nfilesTest-1));

        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
        
        //define architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. 
                //.weightInit(WeightInit.XAVIER)
                //.weightInit(WeightInit.LECUN_NORMAL)
                .weightInit(WeightInit.SIGMOID_UNIFORM)
                .updater(new Nadam())
                //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) 
                //.gradientNormalizationThreshold(0.5)
                .list()
                //.layer(new SubsamplingLayer.Builder(Subsampling1DLayer.PoolingType.AVG).kernelSize(2,1).stride(1,1).build())
                .layer(new LSTM.Builder().activation(Activation.TANH).nIn(nInput).nOut(nhidden).dropOut(0.2).build())
                .layer( new SelfAttentionLayer.Builder().nOut(nhidden).nHeads(1).projectInput(true).build())
                //.layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(nhidden).nOut(numLabelClasses).build())
               //.layer(new OutputLayer.Builder().nOut(2).activation(Activation.SOFTMAX)
                //        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        conf.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        /*
         * .layer(new LSTM.Builder().nOut(8).build())
            .layer( new SelfAttentionLayer.Builder().nOut(4).nHeads(2).projectInput(true).build())
            .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).build())
            .layer(new OutputLayer.Builder().nOut(2).activation(Activation.SOFTMAX)
                    .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                    
         */
        //start learning
        net = new MultiLayerNetworkWhiteBox(conf);
        net.init();
        log.trace(net.summary());
        //System.exit(0);
        log.info("Starting training...");
        net.setListeners(new ScoreIterationListener(20), new EvaluativeListener(testData, 1, InvocationType.EPOCH_END));   //Print the score (loss function value) every 20 iterations
        net.fit(trainData, nEpochs);
        
        //testing performance
        log.info("Evaluating...");
        Evaluation eval = net.evaluate(testData);
        log.trace(eval.stats());
        accuracy = eval.accuracy();
        evaluation = eval.stats();
        //solve(testFile);
        
        //net.save(savedNet); // this operation does not include normalisation - use the following one
        MultiLayerNetworkWhiteBoxSerializer.writeModel(net, savedNormalizer, false, normalizer);
        
        if (new File("./logs/application.log").exists())
        	Files.copy(new File("./logs/application.log"), new File("./logs/",savedNet.getName()+".log"));
        
        
        
        log.info("TRAINING DONE");
        //System.exit(0);
	}
	
	

	public int classify (File testFile) throws Exception {
		double[][] matrix = Utils4Models.file2Matrix(testFile, false);
		return classify(matrix);
	}
	
	public int classify (double[][] matrix) throws Exception {
		
		if (normalizer==null)
			normalizer = ModelSerializer.restoreNormalizerFromFile(savedNormalizer);
		
		SequenceRecordReaderDataSetIterator iterator = Utils4Models.timeSeriesMatrixToIterator(matrix);
		iterator.setPreProcessor(normalizer);
				
		if (net==null)
			net = (MultiLayerNetworkWhiteBox) MultiLayerNetworkWhiteBoxSerializer.restoreMultiLayerNetwork(savedNormalizer);
		
		log.trace("Net file loaded");

		iterator.reset();
		log.trace(net.summary());
        net.output(iterator);
        
        double [] finaloutput = null;
		double [] attentionoutput = null;
		
		int nlayers = net.layerOutputs.size();
		for (int d=0;d<nlayers;d++) {
			
			INDArray outlayer = net.layerOutputs.get(d);
			long [] shape = outlayer.shape();
			long noutput = shape[1];
			long ltimes = shape[2];
			log.trace("Layer "+(d+1));
			for (int t=0;t<ltimes;t++) {
				INDArray output_t = outlayer.slice(t, 2);
				double [] vectorO = output_t.data().getDoublesAt(0, (int)noutput);
				log.trace("Output t"+(t+1)+": "+Arrays.toString(vectorO));
				if (d == (nlayers-1) && t == (ltimes-1))
					finaloutput = vectorO;
				if (d == (nlayers-2) && t == (ltimes-1))
					attentionoutput = vectorO;
			}
		}
		
		double class0 = finaloutput[0];
		double class1 = finaloutput[1];
		
		log.trace("Class 0 score: "+class0);
		log.trace("Class 1 score: "+class1);
		
		int classification = -1;
		if (class0>class1) {
			log.trace("CLASSIFICATION 0");
			classification = 0;
		}
		else {
			log.trace("CLASSIFICATION 1");
			classification = 1;
		}
		log.trace("Attention vector "+Arrays.toString(attentionoutput));
		
		double attentionscore = 0;
		for (double attention:attentionoutput) {
			attentionscore  = attentionscore + attention;
		}
		
		log.trace("ATTENTION SCORE: "+attentionscore);
		currentAttentionScore = attentionscore;
		log.trace("Done!:\n");
		
		return classification;
	}

	public void solve(File testFile) throws Exception {
		
		log.info("Test file loaded");
		double[][] matrix = Utils4Models.file2Matrix(testFile, false);
		long timeSeriesLength = matrix.length; //n rows
		
		if (normalizer==null)
			normalizer = ModelSerializer.restoreNormalizerFromFile(savedNormalizer);
		
		SequenceRecordReaderDataSetIterator iterator = Utils4Models.timeSeriesMatrixToIterator(matrix);
		iterator.setPreProcessor(normalizer);
				
		if (net==null)
			//net = MultiLayerNetwork.load(savedNet, false);
			net = (MultiLayerNetworkWhiteBox) MultiLayerNetworkWhiteBoxSerializer.restoreMultiLayerNetwork(savedNormalizer);
		
		//MultiLayerConfiguration cfg = (MultiLayerConfiguration) net.getConfig();
		//cfg.setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
		
		log.info("Net file loaded");
		

		iterator.reset();
		
		// we create config with 10MB memory space pre allocated
        WorkspaceConfiguration initialConfig = WorkspaceConfiguration.builder()
            .initialSize(10 * 1024L * 1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.NONE)
            .build();

        INDArray result;
        
        //Tests for MemoryWorkspace
        //MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "WS_ID");
        log.trace(net.summary());       
        INDArray output = net.output(iterator);
        
        
        
        //INDArray output = this.output(iterator,false,ws);
		
		log.trace("COMPLETE OUTPUT 0 class "+Arrays.toString(output.data().getDoublesAt(0,matrix.length)));
		log.trace("COMPLETE OUTPUT 1 class "+Arrays.toString(output.data().getDoublesAt(matrix.length,matrix.length*2)));
		
		double [] finaloutput = null;
		double [] attentionoutput = null;
		
		int nlayers = net.layerOutputs.size();
		for (int d=0;d<nlayers;d++) {
			
			INDArray outlayer = net.layerOutputs.get(d);
			long [] shape = outlayer.shape();
			long noutput = shape[1];
			long ltimes = shape[2];
			log.trace("Layer "+(d+1));
			for (int t=0;t<ltimes;t++) {
				INDArray output_t = outlayer.slice(t, 2);
				double [] vectorO = output_t.data().getDoublesAt(0, (int)noutput);
				log.trace("Output t"+(t+1)+": "+Arrays.toString(vectorO));
				
				if (d == (nlayers-1) && t == (ltimes-1))
					finaloutput = vectorO;
				if (d == (nlayers-2) && t == (ltimes-1))
					attentionoutput = vectorO;
			}
			
			
			//log.info("Layer "+(d+1)+" output: "+outlayer.data());
		}
		
		double class0 = finaloutput[0];
		double class1 = finaloutput[1];
		
		log.trace("Class 0 score: "+class0);
		log.trace("Class 1 score: "+class1);
		
		if (class0>class1)
			log.trace("CLASSIFICATION 0");
		else
			log.trace("CLASSIFICATION 1");
		
		
		log.trace("Attention vector "+Arrays.toString(attentionoutput));
		
		double attentionscore = 0;
		
		for (double attention:attentionoutput) {
			
			attentionscore  = attentionscore + attention;
			
		}
		
		log.trace("ATTENTION SCORE: "+attentionscore);
		
		System.exit(0);
			
		//Layer[] allLayers = net.getLayers();
		
		//Layer lastLayer = allLayers[allLayers.length-1];
		
		//MemoryWorkspace mw = Nd4j.getMemoryManager().getCurrentWorkspace();
		
		
		//Map<String,Pointer> helperWorkspaces = net.getHelperWorkspaces();
		log.trace("");
		
		
		
		long nOutputs = output.size(1);
		log.trace("# expected outputs : "+nOutputs);
		log.trace("time series length : "+timeSeriesLength);
		
		long maxClass = 0;
        double maxScore = 0;
        
        for (long k=0;k<nOutputs;k++)
        {
        	double score = output.getDouble(0,k,timeSeriesLength-1);
        	log.trace("Output "+k+" score on last vector: "+score);
        	if (maxScore<score) {
        		maxClass = k;
        		maxScore = score;
        	}
        }
        
        log.trace("Best class:\n"+maxClass);
        
        String detectedLabel = baseTestDir.listFiles()[(int)maxClass].getName();
        log.trace("Best label: "+detectedLabel);
        
        log.trace("##############");
        
        //ATTENTION DISCOVERY
        iterator.reset();
        int it=0;
        while (iterator.hasNext()) {
        	
        	DataSet ds = iterator.next();
        	int nsamples = ds.numExamples();
        	log.trace("n "+nsamples);
        	INDArray f = ds.get(0).getFeatures();
			log.trace("Sample #inputs "+f.shapeInfoToString());
						
			log.trace("1 "+f.size(0));
			log.trace("v "+f.size(1));
			log.trace("t "+f.size(2));
			long tslength = f.size(2);
			
			for (long h=0;h<tslength;h++) {
				
				INDArray column = f.slice(h,2);
				log.trace("input "+Arrays.toString(column.data().asDouble()));
				log.trace("column "+h+": "+column.length());
				INDArray reshapedInput = column.reshape(1,3,1);
				log.trace("rinput "+Arrays.toString(reshapedInput.data().asDouble()));
				INDArray outh = net.rnnTimeStep(reshapedInput);
				
				//wArr = w.data().asDouble().toList
				log.trace("output "+Arrays.toString(outh.data().asDouble()));
				
			}
			
			
        	it++;
        }
        
        log.trace("# datasets : "+it);
        
		
        
		log.trace("Done!:\n");
	}
	
	
	 public INDArray output(DataSetIterator iterator, boolean train, MemoryWorkspace ws) {
	        List<INDArray> outList = new ArrayList<>();
	        long[] firstOutputShape = null;
	        while (iterator.hasNext()) {
	            DataSet next = iterator.next();
	            INDArray features = next.getFeatures();

	            if (features == null)
	                continue;

	            INDArray fMask = next.getFeaturesMaskArray();
	            INDArray lMask = next.getLabelsMaskArray();
	            INDArray output = net.output(features, train, fMask, lMask,ws);
	            outList.add(output);
	            if(firstOutputShape == null){
	                firstOutputShape = output.shape();
	            } else {
	                //Validate that shapes are the same (may not be, for some RNN variable length time series applications)
	                long[] currShape = output.shape();
	                Preconditions.checkState(firstOutputShape.length == currShape.length, "Error during forward pass:" +
	                        "different minibatches have different output array ranks - first minibatch shape %s, last minibatch shape %s", firstOutputShape, currShape);
	                for( int i=1; i<currShape.length; i++ ){    //Skip checking minibatch dimension, fine if this varies
	                    Preconditions.checkState(firstOutputShape[i] == currShape[i], "Current output shape does not match first" +
	                            " output array shape at position %s: all dimensions must match other than the first dimension.\n" +
	                            " For variable length output size/length use cases such as for RNNs with multiple sequence lengths," +
	                            " use one of the other (non iterator) output methods. First batch output shape: %s, current batch output shape: %s",
	                            i, firstOutputShape, currShape);
	                }
	            }
	        }
	        return Nd4j.concat(0, outList.toArray(new INDArray[outList.size()]));
	    }
	 
}
