package it.cnr.speech.acousticmodelling.experiments.lstm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Files;
import java.util.List;

public class PrepareSpeeconData {

	public static void main(String[] args) throws Exception{
		File trainingFolderOrig = new File("D:\\WorkFolder\\Experiments\\Speech Recognition Speecon\\Speecon syllables\\train");
		File trainingFolderDest = new File("D:\\WorkFolder\\Experiments\\LSTMJava\\LongShortTermMemory\\datafiles\\train\\");
		File testFolderOrig = new File("D:\\WorkFolder\\Experiments\\Speech Recognition Speecon\\Speecon syllables\\test");
		File testFolderDest = new File("D:\\WorkFolder\\Experiments\\LSTMJava\\LongShortTermMemory\\datafiles\\test");
		orchestrate(trainingFolderOrig, trainingFolderDest);
		orchestrate(testFolderOrig, testFolderDest);

	}
	public static void orchestrate(File mainfolder, File outputFolder) throws Exception{
		
		File [] allSubFolders = mainfolder.listFiles();
		int classNumber = 0;
		int offset = 0;
		for (File sub:allSubFolders) {
			System.out.println(sub.getName()+"<->"+classNumber);
			File featuresfolder = new File(outputFolder,"features");
			File labelsfolder = new File(outputFolder,"labels");
			offset = transformAll(sub, featuresfolder, labelsfolder, offset, ""+classNumber);
			
			classNumber++;
			//if (classNumber>1)
				//break;
		}
		
	}
	
	
	public static int transformAll(File folder, File outputFolder, File outputFolderLabels, int offset, String className) throws Exception{
		File [] allFiles = folder.listFiles();
		
		for (File f:allFiles) {
			if (f.getName().endsWith(".csv")) {

				File outputFile = new File(outputFolder,offset+".csv");
				File outputLabelFile = new File(outputFolderLabels,offset+".csv");
				
				System.out.println(f.getName()+"-->"+outputFile.getAbsolutePath());
				
				writeClass(className, outputLabelFile);
				transform(f,outputFile);
				offset++;
				
			}
			
		}
		
		return offset;
		
	}
	
	public static void writeClass(String classname,File output) throws Exception{
		BufferedWriter bw = new BufferedWriter(new FileWriter(output));
		bw.write(classname);
		bw.close();
	}
	
	public static void transform(File input,File output) throws Exception{
		
		List<String> all = Files.readAllLines(input.toPath());
		BufferedWriter bw = new BufferedWriter(new FileWriter(output));
		int i = 0;
		for (String s:all) {
			if (i>0) {
				
				bw.write(s);
				
				if (i<all.size()-1)
					bw.write("\n");
				
			}
			i++;
		}
		bw.close();
	}
	
}
