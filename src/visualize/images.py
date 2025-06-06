import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.widgets import Button


def visualize_images(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    rows: int = 3,
    columns: int = 4,
    image_color_map = "gray",
    win_size: tuple[int, int] = (8, 6),
    background_color: str | Colormap | None = None,
) -> None:
    """
    Display all images in one window with multiple pages (rows x columns).
    You can press the left and right arrow keys or the onscreen buttons
    to flip through each page of subplots.
    """

    if (len(images) != len(labels)) or (len(labels) != len(predictions)):
        raise ValueError(
            f"Images, labels and predictions must be the same length. "
            f"Their current lengths are {len(images)}, {len(labels)} "
            f"and {len(predictions)} respectively"
        )

    N = len(images)
    per_page = rows * columns
    n_pages = int(np.ceil(N / per_page))
    current_page = 0

    # Create a single figure + grid of axes once:
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=win_size,
        facecolor=background_color
    )
    # Flatten axes to a 1D array so we can index 0 ... (rows*columns-1)
    axes = axes.ravel()

    # Navigation buttons
    btn_prev_pos = plt.axes((0.3, 0.05, 0.1, 0.04))
    btn_next_pos = plt.axes((0.6, 0.05, 0.1, 0.04))
    btn_prev = Button(btn_prev_pos, "Previous")
    btn_next = Button(btn_next_pos, "Next")

    def draw_page(page_idx: int):
        """
        Clear each Axes and redraw images for page number ``page_idx``
        (0-based).
        """

        # Clear titles & images on all axes first
        for ax in axes:
            ax.clear()

        start = page_idx * per_page
        for slot in range(per_page):
            img_idx = start + slot
            ax = axes[slot]

            # While we still have images to display
            if img_idx < N:
                # If images are flattened (28*28), reshape;
                # otherwise assume already shaped
                img = images[img_idx]
                if img.ndim == 1: img = img.reshape(28, 28)

                ax.imshow(img, cmap=image_color_map)
                ax.set_title(f"True: {labels[img_idx]}, Pred: {predictions[img_idx]}")

            # Whether or not we still have images to display,
            # we turn that Axes off to prevent clutter
            ax.axis("off")

        fig.suptitle(f"Page {page_idx + 1} / {n_pages}", fontsize=16)
        fig.canvas.draw_idle()

    def on_key(event):
        """
        Called when a key is pressed. Use left/right arrow keys to
        decrease/increase the page.
        """

        nonlocal current_page
        if event.key == "right":
            current_page = (current_page + 1) % n_pages
            draw_page(current_page)
        elif event.key == "left":
            current_page = (current_page - 1) % n_pages
            draw_page(current_page)

    def next_page(event):
        nonlocal current_page
        current_page = (current_page + 1) % n_pages
        draw_page(current_page)

    def prev_page(event):
        nonlocal current_page
        current_page = (current_page - 1) % n_pages
        draw_page(current_page)

    # Connect the key‐press event:
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Connect the on-clicked button events:
    btn_next.on_clicked(next_page)
    btn_prev.on_clicked(prev_page)

    # Draw the first page:
    draw_page(current_page)

    plt.show()
