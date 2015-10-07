package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Dan Klein
 * @author Spence Green
 */
public class PMIAligner extends BaseAligner {

  private static final long serialVersionUID = 1315751943476440515L;

  private CounterMap<String,String> sourceTargetCounts;
  private Counter<String> sourceWordCount;
  private Counter<String> targetWordCount;

  @Override
  protected double calculateAlignmentProb(String sourceWord, String targetWord) {
    double p_source = sourceWordCount.getCount(sourceWord) / sourceWordCount.totalCount();
    double p_target = targetWordCount.getCount(targetWord) / targetWordCount.totalCount();

    Counter<String> counterForSource = sourceTargetCounts.getCounter(sourceWord);
    double p_target_given_source = counterForSource.getCount(targetWord) / counterForSource.totalCount();

    return (p_source * p_target_given_source) / (p_source * p_target);
  }

  public void train(List<SentencePair> trainingPairs) {
    sourceTargetCounts = new CounterMap<String, String>();
    sourceWordCount = new Counter<String>();
    targetWordCount = new Counter<String>();

    for(SentencePair pair : trainingPairs) {
      sourceWordCount.incrementAll(pair.getSourceWords(), 1.0);
      targetWordCount.incrementAll(targetWordsWithNull(pair), 1.0);

      for (String sourceWord : pair.getSourceWords()) {
        for (String targetWord : targetWordsWithNull(pair)) {
          sourceTargetCounts.incrementCount(sourceWord, targetWord, 1.0);
        }
      }
    }
  }
}
