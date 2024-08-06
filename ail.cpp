#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iterator>
#include <cassert>

// Hyperparameter settings
const int EMBEDDING_SIZE = 128; // Size of Embedding vector
const int HIDDEN_SIZE = 512; // Size of Hidden layer
const int OUTPUT_SIZE = 128; // Size of Final print
const int NUM_HEADS = 8; // Number of MHA
const int CONTEXT_SIZE = 120; // Size of Context
const double LEARNING_RATE = 0.01; // Learning Rate
const double EPSILON = 1e-10; // Epsilon
const int EPOCH_NUM = 30; // Number of epoch
const double BETA1 = 0.9; // BetaA for Adam
const double BETA2 = 0.999; // BetaB for Adam
const std::string PAD_TOKEN = "<PAD>"; // Pad token

std::vector<std::vector<double>> generateWeightMatrix(int inputSize, int outputSize) {
    std::vector<std::vector<double>> weightMatrix(inputSize, std::vector<double>(outputSize));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0 / std::sqrt(inputSize));
    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weightMatrix[i][j] = dis(gen);
        }
    }
    return weightMatrix;
}

std::vector<double> transformEmbedding(const std::vector<double>& inputEmbedding,
    const std::vector<std::vector<double>>& weightMatrix) {
    std::vector<double> transformedEmbedding(weightMatrix[0].size(), 0.0);
    for (size_t j = 0; j < weightMatrix[0].size(); ++j) {
        for (size_t i = 0; i < inputEmbedding.size(); ++i) {
            transformedEmbedding[j] += inputEmbedding[i] * weightMatrix[i][j];
        }
    }
    return transformedEmbedding;
}

std::vector<double> generateRandomEmbedding() {
    std::vector<double> embedding(EMBEDDING_SIZE);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < EMBEDDING_SIZE; ++i) {
        embedding[i] = dis(gen);
    }
    return embedding;
}

double cosineSimilarity(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        normA += vec1[i] * vec1[i];
        normB += vec2[i] * vec2[i];
    }
    return dotProduct / (std::sqrt(normA + EPSILON) * std::sqrt(normB + EPSILON));
}

std::vector<std::string> tokenize(const std::string& sentence) {
    std::istringstream stream(sentence);
    std::vector<std::string> tokens((std::istream_iterator<std::string>(stream)),
        std::istream_iterator<std::string>());
    return tokens;
}

std::vector<double> aggregateEmbeddings(const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, std::vector<double>>& embeddings) {
    std::vector<double> aggregated(EMBEDDING_SIZE, 0.0);
    for (const auto& token : tokens) {
        auto it = embeddings.find(token);
        if (it != embeddings.end()) {
            for (size_t i = 0; i < EMBEDDING_SIZE; ++i) {
                aggregated[i] += it->second[i];
            }
        }
        else if (embeddings.find(PAD_TOKEN) != embeddings.end()) {
            for (size_t i = 0; i < EMBEDDING_SIZE; ++i) {
                aggregated[i] += embeddings.at(PAD_TOKEN)[i];
            }
        }
    }
    for (auto& value : aggregated) {
        value /= tokens.size();
    }
    return aggregated;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    std::vector<double> exps(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        exps[i] = std::exp(logits[i]);
        sum += exps[i];
    }
    for (auto& exp : exps) {
        exp /= sum;
    }
    return exps;
}

double crossEntropyLoss(const std::vector<double>& predicted, const std::vector<double>& target) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        loss -= target[i] * std::log(predicted[i] + EPSILON);
    }
    return loss;
}

void normalizeEmbedding(std::vector<double>& embedding) {
    double norm = std::sqrt(std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0));
    for (auto& value : embedding) {
        value /= (norm + EPSILON);
    }
}

std::vector<double> selfAttention(const std::vector<double>& query, const std::vector<double>& key, const std::vector<double>& value) {
    double score = std::inner_product(query.begin(), query.end(), key.begin(), 0.0);
    score = std::exp(score / std::sqrt(EMBEDDING_SIZE));
    std::vector<double> attentionOutput(EMBEDDING_SIZE);
    for (size_t i = 0; i < EMBEDDING_SIZE; ++i) {
        attentionOutput[i] = score * value[i];
    }
    return attentionOutput;
}

std::vector<double> multiHeadAttention(const std::vector<double>& query,
    const std::vector<std::vector<double>>& keys,
    const std::vector<std::vector<double>>& values) {
    std::vector<double> multiHeadOutput(EMBEDDING_SIZE, 0.0);
    int headSize = EMBEDDING_SIZE / NUM_HEADS;
    for (int h = 0; h < NUM_HEADS; ++h) {
        std::vector<double> headOutput(headSize, 0.0);
        for (size_t i = 0; i < keys.size(); ++i) {
            auto attentionOutput = selfAttention(query, keys[i], values[i]);
            for (int j = 0; j < headSize; ++j) {
                int index = h * headSize + j;
                if (index < attentionOutput.size()) {
                    headOutput[j] += attentionOutput[index];
                }
            }
        }
        for (int j = 0; j < headSize; ++j) {
            multiHeadOutput[h * headSize + j] = headOutput[j];
        }
    }
    return multiHeadOutput;
}

std::string predictNextToken(const std::vector<double>& contextEmbedding,
    const std::unordered_map<std::string, std::vector<double>>& embeddings) {
    double maxSimilarity = -1.0;
    std::string nextToken;
    for (const auto& pair : embeddings) {
        double similarity = cosineSimilarity(contextEmbedding, pair.second);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            nextToken = pair.first;
        }
    }
    return nextToken;
}

void padAndMask(std::vector<std::string>& tokens, int contextSize) {
    if (tokens.size() < contextSize) {
        tokens.insert(tokens.end(), contextSize - tokens.size(), PAD_TOKEN);
    }
}

void trainModel(std::unordered_map<std::string, std::vector<double>>& embeddings,
    const std::vector<std::string>& trainingData) {
    std::unordered_map<std::string, unsigned int> tokenToIndex;
    for (size_t i = 0; i < trainingData.size(); ++i) {
        tokenToIndex[trainingData[i]] = i;
    }

    auto weightMatrix = generateWeightMatrix(HIDDEN_SIZE, OUTPUT_SIZE);

    std::unordered_map<std::string, std::vector<double>> m, v;
    for (const auto& token : trainingData) {
        m[token] = std::vector<double>(EMBEDDING_SIZE, 0.0);
        v[token] = std::vector<double>(EMBEDDING_SIZE, 0.0);
    }

    for (int epoch = 0; epoch < EPOCH_NUM; ++epoch) {
        double totalLoss = 0.0;
        for (size_t i = 0; i < trainingData.size() - CONTEXT_SIZE; ++i) {
            std::vector<std::string> context(trainingData.begin() + i, trainingData.begin() + i + CONTEXT_SIZE);
            padAndMask(context, CONTEXT_SIZE);

            if (i + CONTEXT_SIZE < trainingData.size()) {
                std::string correctToken = trainingData[i + CONTEXT_SIZE];
                std::vector<double> contextEmbedding = aggregateEmbeddings(context, embeddings);
                std::vector<double> correctEmbedding = embeddings[correctToken];

                auto transformedContextEmbedding = transformEmbedding(contextEmbedding, weightMatrix);
                auto multiHeadOutput = multiHeadAttention(transformedContextEmbedding, { transformedContextEmbedding }, { transformedContextEmbedding });
                normalizeEmbedding(multiHeadOutput);

                auto predictedEmbedding = softmax(multiHeadOutput);

                std::vector<double> correctEmbeddingOneHot(predictedEmbedding.size(), 0.0);
                auto it = tokenToIndex.find(correctToken);
                if (it != tokenToIndex.end() && it->second < correctEmbeddingOneHot.size()) {
                    correctEmbeddingOneHot[it->second] = 1.0;
                }

                double loss = crossEntropyLoss(predictedEmbedding, correctEmbeddingOneHot);
                totalLoss += loss;

                for (size_t j = 0; j < EMBEDDING_SIZE; ++j) {
                    double g = (predictedEmbedding[j] - correctEmbeddingOneHot[j]);
                    m[correctToken][j] = BETA1 * m[correctToken][j] + (1 - BETA1) * g;
                    v[correctToken][j] = BETA2 * v[correctToken][j] + (1 - BETA2) * g * g;

                    double m_hat = m[correctToken][j] / (1 - pow(BETA1, epoch + 1));
                    double v_hat = v[correctToken][j] / (1 - pow(BETA2, epoch + 1));

                    embeddings[correctToken][j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
                }

                normalizeEmbedding(embeddings[correctToken]);
            }
        }
        std::cout << "Epoch " << epoch + 1 << " Loss: " << totalLoss / trainingData.size() << "\n    [ ===================== " << epoch+1 << " / " << EPOCH_NUM << " ]" << "\n";
    }
}

void evaluateModel(const std::unordered_map<std::string, std::vector<double>>& embeddings,
    const std::vector<std::string>& testData) {
    int correct = 0;
    for (size_t i = 0; i < testData.size() - CONTEXT_SIZE; ++i) {
        std::vector<std::string> context(testData.begin() + i, testData.begin() + i + CONTEXT_SIZE);
        padAndMask(context, CONTEXT_SIZE);

        if (i + CONTEXT_SIZE < testData.size()) {
            std::string correctToken = testData[i + CONTEXT_SIZE];
            std::vector<double> contextEmbedding = aggregateEmbeddings(context, embeddings);
            std::string predictedToken = predictNextToken(contextEmbedding, embeddings);
            if (predictedToken == correctToken) {
                correct++;
            }
        }
    }

        double accuracy = static_cast<double>(correct) / (testData.size() - CONTEXT_SIZE) * 100;
        std::cout << "Accuracy: " << accuracy << "%\n";
}

int main() {
    std::vector<std::string> trainingData = {
        "In", "a", "small", "village", "nestled", "between", "rolling", "hills", "there", "lived", "a", "young", "woman", "named", "Elara",
        "Elara", "was", "known", "for", "her", "curiosity", "and", "love", "for", "exploring", "the", "unknown", "One", "day", "while",
        "wandering", "through", "the", "forest", "she", "stumbled", "upon", "an", "old", "rusted", "key", "lying", "on", "the", "ground",
        "It", "was", "unlike", "anything", "she", "had", "ever", "seen", "with", "intricate", "engravings", "that", "seemed", "to",
        "tell", "a", "story", "Intrigued", "Elara", "took", "the", "key", "and", "began", "to", "search", "for", "what", "it",
        "might", "unlock", "She", "asked", "the", "village", "elders", "but", "no", "one", "knew", "of", "any", "door", "or",
        "chest", "that", "required", "such", "a", "key", "Determined", "to", "uncover", "its", "secret", "she", "decided", "to",
        "venture", "deeper", "into", "the", "forest", "to", "places", "she", "had", "never", "dared", "to", "go", "before", "As",
        "she", "journeyed", "deeper", "the", "forest", "became", "denser", "and", "the", "air", "grew", "cooler", "After", "hours",
        "of", "walking", "she", "came", "upon", "a", "forgotten", "stone", "structure", "hidden", "behind", "thick", "vines", "and",
        "moss", "At", "the", "center", "of", "the", "structure", "was", "a", "massive", "ancient", "door", "adorned", "with",
        "carvings", "that", "matched", "the", "ones", "on", "the", "key", "With", "a", "mix", "of", "excitement", "and", "apprehension",
        "Elara", "inserted", "the", "key", "into", "the", "lock", "and", "turned", "it", "The", "door", "creaked", "open", "to",
        "reveal", "a", "hidden", "room", "filled", "with", "relics", "and", "treasures", "from", "a", "bygone", "era", "But",
        "what", "caught", "Elara's", "attention", "was", "a", "large", "ornate", "mirror", "standing", "at", "the", "room's",
        "center", "As", "she", "approached", "the", "mirror", "shimmered", "and", "showed", "not", "her", "reflection", "but",
        "a", "different", "world", "one", "filled", "with", "vibrant", "colors", "floating", "islands", "and", "creatures", "that",
        "defied", "description", "Curiosity", "piqued", "Elara", "reached", "out", "to", "touch", "the", "mirror", "To", "her",
        "astonishment", "her", "hand", "passed", "through", "the", "glass", "followed", "by", "the", "rest", "of", "her", "body",
        "She", "found", "herself", "in", "the", "strange", "magical", "world", "she", "had", "seen", "in", "the", "mirror",
        "Here", "she", "met", "beings", "who", "spoke", "in", "riddles", "and", "witnessed", "wonders", "she", "could", "never",
        "have", "imagined", "She", "learned", "that", "the", "key", "she", "had", "found", "was", "a", "portal", "between",
        "worlds", "a", "link", "to", "a", "realm", "of", "infinite", "possibilities", "Elara", "spent", "many", "days", "exploring",
        "this", "new", "world", "making", "friends", "and", "uncovering", "its", "mysteries", "But", "as", "much", "as", "she",
        "loved", "the", "adventure", "she", "knew", "she", "couldn't", "stay", "forever", "With", "a", "heavy", "heart", "she",
        "returned", "to", "the", "mirror", "and", "stepped", "back", "into", "her", "own", "world", "The", "door", "closed",
        "behind", "her", "and", "the", "key", "turned", "to", "dust", "in", "her", "hand", "Though", "the", "portal", "was",
        "gone", "Elara's", "life", "was", "forever", "changed", "She", "returned", "to", "her", "village", "with", "stories",
        "of", "her", "incredible", "journey", "and", "a", "newfound", "appreciation", "for", "the", "magic", "in", "the", "everyday",
        "world", "And", "though", "she", "never", "found", "another", "key", "like", "the", "one", "she", "lost", "she", "always",
        "kept", "her", "eyes", "open", "wondering", "if", "someday", "somewhere", "she", "might", "find", "another", "door",
        "to", "another", "world"
    }; // Training data example

    std::vector<std::string> testData = {
        "Elara", "spent", "many", "days", "exploring", "this", "new", "world", "making", "friends", "and", "uncovering", "its", "mysteries",
        "But", "as", "much", "as", "she", "loved", "the", "adventure", "she", "knew", "she", "couldn't", "stay", "forever", "With",
        "a", "heavy", "heart", "she", "returned", "to", "the", "mirror", "and", "stepped", "back", "into", "her", "own", "world",
        "The", "door", "closed", "behind", "her", "and", "the", "key", "turned", "to", "dust", "in", "her", "hand", "Though",
        "the", "portal", "was", "gone", "Elara's", "life", "was", "forever", "changed", "She", "returned", "to", "her", "village",
        "with", "stories", "of", "her", "incredible", "journey", "and", "a", "newfound", "appreciation", "for", "the", "magic", "in",
        "the", "everyday", "world", "And", "though", "she", "never", "found", "another", "key", "like", "the", "one", "she", "lost",
        "she", "always", "kept", "her", "eyes", "open", "wondering", "if", "someday", "somewhere", "she", "might", "find", "another",
        "door", "to", "another", "world"
    }; // Test data example

    std::unordered_map<std::string, std::vector<double>> embeddings;
    for (const auto& word : trainingData) {
        embeddings[word] = generateRandomEmbedding();
    }
    for (const auto& word : testData) {
        if (embeddings.find(word) == embeddings.end()) {
            embeddings[word] = generateRandomEmbedding();
        }
    }

    embeddings[PAD_TOKEN] = std::vector<double>(EMBEDDING_SIZE, 0.0);

    trainModel(embeddings, trainingData);

    evaluateModel(embeddings, testData);

    std::string inputSentence;
    std::cout << "Enter a sentence: ";
    std::getline(std::cin, inputSentence);

    std::vector<std::string> tokens = tokenize(inputSentence);
    padAndMask(tokens, CONTEXT_SIZE);
    std::vector<double> contextEmbedding = aggregateEmbeddings(tokens, embeddings);
    std::string nextToken = predictNextToken(contextEmbedding, embeddings);

    std::cout << "Predicted next token: " << nextToken << "\n";

    return 0;
}
