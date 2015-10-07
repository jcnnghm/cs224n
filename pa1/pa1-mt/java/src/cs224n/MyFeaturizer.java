package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.util.ArrayList;
import java.util.Properties;

import edu.stanford.nlp.mt.util.FeatureValue;
import edu.stanford.nlp.mt.util.Featurizable;
import edu.stanford.nlp.mt.util.IString;
import edu.stanford.nlp.mt.decoder.feat.RuleFeaturizer;
import edu.stanford.nlp.util.ArrayCoreMap;
import edu.stanford.nlp.util.Generics;
import edu.stanford.nlp.mt.util.TokenUtils;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.*;
import edu.stanford.nlp.ling.*;

/**
 * A rule featurizer.
 */
public class MyFeaturizer implements RuleFeaturizer<IString, String> {

  private static final String FEATURE_NAME = "SIZE_CHANGE";
  private StanfordCoreNLP pipeline;

  @Override
  public void initialize() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos");
    pipeline = new StanfordCoreNLP(props);
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    List<FeatureValue<String>> features = Generics.newLinkedList();
    features.add(new FeatureValue<String>("bias", 1.0));

    List<String> posTags = getPosTags(f.targetPhrase.toString());
    for (String pos : posTags) {
      features.add(new FeatureValue<String>(String.format("POS:%s", pos), 1.0));
    }

    for (int i=-2; i<posTags.size(); i++) {
      features.add(new FeatureValue<String>(String.format("DONE_POS:%s:%s:%s", getPos(posTags, i), getPos(posTags, i+1), getPos(posTags, i+2)), 1.0));
    }

    /*
    int targetSize = f.targetPhrase.size();
    int sourceSize = f.sourcePhrase.size();

    features.add(new FeatureValue<String>(String.format("%s:%d", FEATURE_NAME, targetSize - sourceSize), 1.0));

    int punctuation = 0;
    for (IString token : f.sourcePhrase) {
      if (TokenUtils.isPunctuation(token.toString())) {
        punctuation++;
      }
    }
    for (IString token : f.targetPhrase) {
      if (TokenUtils.isPunctuation(token.toString())) {
        punctuation--;
      }
    }
    features.add(new FeatureValue<String>(String.format("PUNCUATION_BALANCE:%d", punctuation), 1.0));
    */

    return features;
  }

  private String getPos(List<String> posTags, int index) {
    if (index < 0) return "NULL";
    if (index >= posTags.size()) return "NULL";
    return posTags.get(index);
  }

  private List<String> getPosTags(String phrase) {
    ArrayList<String> partsOfSpeech = new ArrayList<String>();
    Annotation document = new Annotation(phrase);
    pipeline.annotate(document);
    List<CoreMap> sentences = document.get(CoreAnnotations.SentencesAnnotation.class);
    for(CoreMap sentence: sentences) {
      for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
        String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
        partsOfSpeech.add(pos);
      }
    }
    return partsOfSpeech;
  }

  @Override
  public boolean isolationScoreOnly() {
    return false;
  }
}
