chosen_topic = [
        "abortion",
        "environment",
        "guns",
        "health care",
        "immigration",
        "LGBTQ",
        "racism",
        "taxes",
        "technology",
        "trade",
        "trump impeachment",
        "us military",
        "us 2020 election",
        "welfare",
    ]

for i in range(len(classes)):
    cl = classes[i]
    print(f'{cl}: ')
    
    arr = recommendation_stats[i]
    arr2 = original_interaction_stats[i]
    
    total = np.sum(arr2.flatten())
    arr2 /= total
    # chosen_topics = chosen_topics.reshape((14, 5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), constrained_layout=True)

    im1 = axes[0].imshow(arr, cmap='Blues', interpolation='none')
    axes[0].set_title(f"Topic Cov: {user_metrics[i]['topic_hit']}, Div: {user_metrics[i]['diversity']}")
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xticklabels([-2, -1, 0, 1, 2])
    axes[0].set_yticks(np.arange(len(chosen_topic)))
    axes[0].set_yticklabels()

    im2 = axes[1].imshow(arr2, cmap='Blues', interpolation='none')
    axes[1].set_title("User Interest relative to Ratings")
    axes[1].set_xticks(np.arange(5))
    axes[1].set_xticklabels([-2, -1, 0, 1, 2])
    axes[1].set_yticks(np.arange(len(chosen_topic)))
    axes[1].set_yticklabels([''] * len(chosen_topic))

    fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
    fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)

    plt.savefig(f'src/data/baseline_data/graphs/{cl}.png')
    plt.close(fig)