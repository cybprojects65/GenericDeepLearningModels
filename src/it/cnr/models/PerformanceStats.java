package it.cnr.models;

import java.io.File;
import java.nio.file.Files;
import java.util.List;

public class PerformanceStats {

	public static void main(String[] args) throws Exception {
		//File log = new File("./logs/net_100hidden_40epo_1000batch_44class.log");
		//File log = new File("./logs/net_100hidden_40epo_100batch_44class.log");
		//File log = new File("./logs/net_200hidden_50epo_1000batch_44class.log");
		//File log = new File("./logs/net_50hidden_100epo_1000batch_44class.log");
		//File log = new File("./logs/net_300hidden_50epo_500batch_44class.log");
		//File log = new File("./logs/net_300hidden_50epo_10batch_44class.log");
		//File log = new File("./logs/temp.txt");
		//File log = new File("./logs/net_250hidden_50epo_200batch_44class.log");
		File log = new File("./logs/net_200hidden_85epo_1000batch_44class.bin.log");
		
		calcPerformanceStats(log);
	}
	
	
	public static void calcPerformanceStats(File logFile) throws Exception{
		
		List<String> allLines = Files.readAllLines(logFile.toPath());
		int lineIdx = 0;
		System.out.println(logFile.getName().replace(".log", "").replace("_", " ").replace("net", "").trim());
		while (lineIdx<allLines.size()) {
			String line = allLines.get(lineIdx);
				
			String toSearch = "Starting evaluation nr.";
			int idx = line.indexOf(toSearch);
			if (line.contains(toSearch)) {
				String nEval = line.substring(toSearch.length()).trim();
				//System.out.println("Eval N."+nEval);
				String toSearch2 = "Accuracy:";
				while (!line.contains(toSearch2)) {
					lineIdx++;
					line = allLines.get(lineIdx);
				}
				String accuracy = line.substring(line.indexOf(toSearch2)+toSearch2.length()+1).trim().replace(",", ".");
				//System.out.println(nEval+"\t"+accuracy);
				System.out.println(accuracy);
			}
			
			
			lineIdx++;
		}
		
	}
	
	
}
