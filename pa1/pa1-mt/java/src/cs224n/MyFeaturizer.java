package edu.stanford.nlp.mt.decoder.feat;

import java.util.List;
import java.util.ArrayList;
import java.util.Properties;
import java.util.TreeSet;

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
  private StanfordCoreNLP frenchPipeline;

  @Override
  public void initialize() {
    Properties props = new Properties();
    props.setProperty("annotators", "tokenize, ssplit, pos");
    pipeline = new StanfordCoreNLP(props);

    Properties frenchProps = new Properties();
    frenchProps.setProperty("annotators", "tokenize, ssplit, pos");
    frenchProps.setProperty("pos.model", "french.tagger");
    frenchPipeline = new StanfordCoreNLP(frenchProps);
  }

  @Override
  public List<FeatureValue<String>> ruleFeaturize(
      Featurizable<IString, String> f) {

    List<FeatureValue<String>> features = Generics.newLinkedList();

    List<String> posTags = getPosTags(pipeline, f.targetPhrase.toString());
    List<String> frenchPosTags = getPosTags(frenchPipeline, f.sourcePhrase.toString());

    for (int i=-2; i<posTags.size(); i++) {
      if (i>=0) features.add(new FeatureValue<String>(String.format("TARGET_POS:%s", getPos(posTags, i)), 1.0));
      if (i>=-1) features.add(new FeatureValue<String>(String.format("TARGET_POS_BIGRAM:%s:%s", getPos(posTags, i), getPos(posTags, i+1)), 1.0));
      features.add(new FeatureValue<String>(String.format("TARGET_POS_TRIGRAM:%s:%s:%s", getPos(posTags, i), getPos(posTags, i+1), getPos(posTags, i+2)), 1.0));
    }

    TreeSet<String> englishTags = new TreeSet<String>(posTags);
    TreeSet<String> frenchTags = new TreeSet<String>(frenchPosTags);

    for (String englishTag : englishTags) {
      if (frenchTags.contains(englishTag)) {
        features.add(new FeatureValue<String>(String.format("POS_MATCH:%s", englishTag), 1.0));
      } else {
        features.add(new FeatureValue<String>(String.format("POS_ETOF_MISMATCH:%s", englishTag), 1.0));
      }
    }
    for (String frenchTag : frenchTags) {
      if (!englishTags.contains(frenchTag)) {
        features.add(new FeatureValue<String>(String.format("POS_FTOE_MISMATCH:%s", frenchTag), 1.0));
      }
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

  private List<String> getPosTags(StanfordCoreNLP pipeline, String phrase) {
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
