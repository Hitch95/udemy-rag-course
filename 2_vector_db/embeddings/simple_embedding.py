import numpy as np
import matplotlib.pyplot as plt
import os

# Let's create simple 2D embeddings for a few words
word_embeddings = {
    "king": [0.99, 0.99],
    "queen": [0.95, 0.98],
    "man": [0.75, 0.50],
    "woman": [0.70, 0.48],
    "cat": [-0.85, 0.55],
    "dog": [-0.80, 0.52]
}

# Plotting the embeddings
plt.figure(figsize=(10, 8))
for word, embedding in word_embeddings.items():
    plt.scatter(embedding[0], embedding[1], label=word)
    plt.annotate(word, (embedding[0], embedding[1]))

plt.title("Simple 2D Word Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
# plt.show()

# Sauvegarder l'image et l'ouvrir
output_path = os.path.join(os.path.dirname(__file__), "embeddings_plot.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Graphique sauvegardé : {output_path}")

# Print out the embeddings
for word, embedding in word_embeddings.items():
    print(f"{word}: {embedding}")