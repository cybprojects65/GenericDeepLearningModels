package it.cnr.models;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionSequenceRecordReader;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Utils4Models {

	private static final Logger log = LoggerFactory.getLogger(Utils4Models.class);
	
	public static void movingAverage(DataSetIterator data, int n) {
		data.reset();
		
		while (data.hasNext()) {
			DataSet dataSet = data.next();
			int nsamples = dataSet.numInputs();
			//INDArray inputArray = dataSet.getFeatures();
			log.info("NInput "+nsamples);
			log.info("NSAMPLES "+dataSet.numExamples());
			log.info("Noutcomes "+dataSet.numOutcomes());
			for (int k = 0; k < nsamples; k++) {
				DataSet r = dataSet.get(k);
				INDArray row = r.getFeatures();
				INDArray rowSlice = row.slice(k, 2);
				int ninput = r.numInputs();
				log.info("NSAMPLES in example "+k+": "+ninput);
				double[] avg = new double[ninput-1];
				for (int y = 0; y<ninput-1;y++) {
					log.info(y+": "+rowSlice.getDouble(y));
					avg[y]=(rowSlice.getDouble(y)+rowSlice.getDouble(y+1));
				}
				
				rowSlice.data().setData(avg);
				//row.putSlice(k, rowSlice);
				
			    //log.info("Columns "+row.columns());
				//TimeSeriesUtils.movingAverage(row, 2);
				
				log.info("averaged sample # "+k);
			}
			
		
		}
	}
	
	public static void showData(DataSetIterator data) {
		data.reset();
		log.info("Showing data..");
		int i = 0;
		//there will be as many datasets as minibatch division
		//for each dataset there will be a number of sequences equal to the # of minibatches
		//one sample corresponds to one sequence
		//one seq is in the form 1 x #features x #sequence lengths
		//thus, the seq will be with rows = #features and columns = #seq length. The opposite of what is in the file.
		while (data.hasNext()) {
			DataSet dataSet = data.next();
			int nsamples = dataSet.numExamples();
			//INDArray inputArray = dataSet.getFeatures();
			log.info("DATA SET #"+i);
			log.info("Vec length: "+nsamples);
			log.info("# sample time series in the training set: "+dataSet.numExamples());
			log.info("Output labels: "+dataSet.numOutcomes());
			
			for (int k = 0; k < nsamples; k++) {
				INDArray f = dataSet.get(k).getFeatures();
				log.info("Sample #"+k);
				log.info("Sample #inputs "+f.shapeInfoToString());
				
				String matrix = dataSet.get(k).getFeatures().toString();
				log.info(""+matrix);
				/*
				DataSet r = dataSet.get(k);
				INDArray row = r.getFeatures();
				row = row.slice(k, 2);
				int ninput = r.numInputs();
				log.info("NSAMPLES in example "+k+": "+ninput);
				for (int y = 0; y<ninput;y++) {
					log.info(y+": "+row.getDouble(y));
				}
				*/
			}
			
		i++;
		}
		
		log.info("...");
	}
	
	public static void printDataSet(DataSetIterator trainData) {
		trainData.reset();
		int i = 0;
		while (trainData.hasNext()) {
			DataSet data = trainData.next();
			// System.out.println("Train data "+i+" - Labels: "+data.getLabels());
			System.out.println("Train data #" + i + ";\n" + "N. Inputs:" + data.numInputs() + ";\n" + "N. Examples:"
					+ data.numExamples() + ";\n" + "N. Outputs:" + data.numOutcomes() + ";\n"

			);

			for (int k = 0; k < data.numExamples(); k++) {
				DataSet ex = data.get(k);
				System.out.println("Example #" + k + ";\n" + "N. Inputs:" + ex.numInputs() + ";\n" + "N. Examples:"
						+ ex.numExamples() + ";\n" + "N. Outputs:" + ex.numOutcomes() + ";\n");

				NDArray inputArray = (NDArray) ex.getFeatures();
				// String array = inputArray.toString();

				System.out.println("Input sample dimension:\n" + "N. Rows:" + inputArray.size(0) + ";\nN. Columns:"
						+ inputArray.size(1) + ";\nArray:\n" + inputArray);
				System.out.println("Outcome:\n" + ex.outcome());
				System.out.println("Outcome Labels:\n" + ex.getLabels());

			}

			i++;
		}
	}

	public static double[][] file2Matrix(File featuresFile, boolean hasHeader) throws Exception {

		List<String> allLines = Files.readAllLines(featuresFile.toPath());
		int nLines = allLines.size();

		int nRows = hasHeader ? (nLines - 1) : nLines;
		int nCols = allLines.get(0).split(",").length;
		int start = hasHeader ? 1 : 0;
		double[][] matrix = new double[nRows][nCols];

		for (int r = start; r < nLines; r++) {
			String[] elements = allLines.get(r).split(",");
			for (int c = 0; c < elements.length; c++) {
				Double e = Double.parseDouble(elements[c]);
				int r1 = hasHeader ? r - 1 : r;
				matrix[r1][c] = e;
			}
		}

		return matrix;
	}

	// one row per feature vector in the matrix
	public static SequenceRecordReaderDataSetIterator timeSeriesMatrixToIterator(double[][] featuresMatrix) {

		Collection<List<Writable>> sequence = new ArrayList<>();
		int nrows = featuresMatrix.length;
		int ncols = featuresMatrix[0].length;

		for (int i = 0; i < nrows; i++) {
			List<Writable> vector = new ArrayList<Writable>();
			for (int j = 0; j < ncols; j++) {
				vector.add((Writable) new DoubleWritable(featuresMatrix[i][j]));
			}
			sequence.add(vector);
		}

		List<Collection<List<Writable>>> testOverSeq = new ArrayList();
		testOverSeq.add(sequence);
		SequenceRecordReader testFeatures = new CollectionSequenceRecordReader(testOverSeq);
		SequenceRecordReaderDataSetIterator iter = new SequenceRecordReaderDataSetIterator(testFeatures, 1, -1, -1,
				false);
		iter.reset();
		return iter;

	}

}
