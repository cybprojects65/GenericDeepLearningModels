package it.cnr.speech.acousticmodelling.experiments.lstm;

import java.io.File;

import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;

import it.cnr.models.Utils4Models;

public class TestLSTM {

    private static File baseDir = new File("D:\\WorkFolder\\Experiments\\Speech Recognition Speecon\\Speecon syllables\\");
    private static File baseTestDir = new File(baseDir, "test");
    
	public static void main(String [] args) throws Exception{
		
		int classToTest = 25;
		int fileToTest = 10;
		
		File testFile = baseTestDir.listFiles()[classToTest].listFiles()[fileToTest];
		String expectedLabel = testFile.getParentFile().getName();
		
		//File netFile = new File("net_100hidden_40epo_1000batch_44class.bin");
		File netFile = new File("net_200hidden_50epo_1000batch_44class.bin");
		
		
		File normalizerFile = new File("Normalizer.bin");
		DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(normalizerFile);
		
		System.out.println("Test file taken "+testFile);
		SequenceRecordReaderDataSetIterator iterator = Utils4Models.timeSeriesMatrixToIterator(Utils4Models.file2Matrix(testFile, true));
		iterator.setPreProcessor(normalizer);
		
		MultiLayerNetwork net = MultiLayerNetwork.load(netFile, false);
		INDArray output = net.output(iterator);
		System.out.println("Output : "+output.size(1));
		
		long nOutputs = output.size(1);
		iterator.reset();
		long timeSeriesLength = iterator.next().getFeatures().size(2);
		
		System.out.println("time series length : "+timeSeriesLength);
		
		long maxClass = 0;
        double maxScore = 0;
        for (long k=0;k<nOutputs;k++)
        {
        	double score = output.getDouble(0,k,timeSeriesLength-1);
        	System.out.println("Exit "+k+" "+score);
        	if (maxScore<score) {
        		maxClass = k;
        		maxScore = score;
        	}
        }
        System.out.println("Best class:\n"+maxClass);
        String detectedLabel = baseTestDir.listFiles()[(int)maxClass].getName();
        System.out.println("Expected: "+expectedLabel);
        System.out.println("Detected: "+detectedLabel);
        //System.out.println("Output:\n"+output);
        
		
		
	}
	
	
	
}
