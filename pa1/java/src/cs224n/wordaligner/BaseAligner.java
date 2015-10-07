package cs224n.wordaligner;

import cs224n.util.Counter;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jcnnghm on 10/3/15.
 */
public abstract class BaseAligner implements WordAligner {
  public Alignment align(SentencePair sentencePair) {
    // Placeholder code below.
    // TODO Implement an inference algorithm for Eq.1 in the assignment
    // handout to predict alignments based on the counts you collected with train().
    Alignment alignment = new Alignment();

    int sourceIndex = 0;
    for (String sourceWord : sentencePair.getSourceWords()) {
      Counter<Integer> alignments = new Counter<Integer>();
      int targetIndex = 0;
      for (String targetWord : targetWordsWithNull(sentencePair)) {
        double pAlignment = calculateAlignmentProb(sourceWord, targetWord, sourceIndex, targetIndex, sentencePair);
        alignments.setCount(targetIndex, pAlignment);
        targetIndex++;
      }

      int alignedTargetIndex = alignments.argMax();
      if (alignedTargetIndex >= 0 && alignedTargetIndex < sentencePair.getTargetWords().size()) {
        alignment.addPredictedAlignment(alignedTargetIndex, sourceIndex);
      }

      sourceIndex++;
    }

    return alignment;
  }

  public static List<String> targetWordsWithNull(SentencePair pair) {
    ArrayList<String> targetWords = new ArrayList<String>(pair.getTargetWords());
    targetWords.add(WordAligner.NULL_WORD);
    return targetWords;
  }

  protected double calculateAlignmentProb(String sourceWord, String targetWord, int sourceIndex, int targetIndex, SentencePair pair) {
    return calculateAlignmentProb(sourceWord, targetWord);
  }

  protected double calculateAlignmentProb(String sourceWord, String targetWord) {
    throw new NotImplementedException();
  }
}
