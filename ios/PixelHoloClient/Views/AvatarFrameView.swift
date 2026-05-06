import Combine
import SwiftUI
import UIKit

struct AvatarFrameView: UIViewRepresentable {
    let player: AvatarPlayer
    var cornerRadius: CGFloat = 20

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> FrameContainerView {
        let view = FrameContainerView()
        view.update(cornerRadius: cornerRadius)
        context.coordinator.bind(player: player, view: view)
        return view
    }

    func updateUIView(_ uiView: FrameContainerView, context: Context) {
        uiView.update(cornerRadius: cornerRadius)
        context.coordinator.bind(player: player, view: uiView)
    }

    final class Coordinator {
        private var cancellable: AnyCancellable?
        private weak var boundView: FrameContainerView?
        private weak var boundPlayer: AvatarPlayer?

        @MainActor
        func bind(player: AvatarPlayer, view: FrameContainerView) {
            guard boundPlayer !== player || boundView !== view else { return }
            cancellable?.cancel()
            boundPlayer = player
            boundView = view
            view.setImage(player.currentFrame)
            cancellable = player.$currentFrame
                .receive(on: DispatchQueue.main)
                .sink { [weak view] image in
                    Task { @MainActor in
                        view?.setImage(image)
                    }
                }
        }
    }
}

final class FrameContainerView: UIView {
    private let imageView = UIImageView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        clipsToBounds = true
        addSubview(imageView)
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.backgroundColor = .clear
        imageView.contentMode = .scaleAspectFit
        imageView.clipsToBounds = true

        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: topAnchor),
            imageView.bottomAnchor.constraint(equalTo: bottomAnchor),
            imageView.leadingAnchor.constraint(equalTo: leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func update(cornerRadius: CGFloat) {
        if layer.cornerRadius != cornerRadius {
            layer.cornerRadius = cornerRadius
            layer.cornerCurve = .continuous
        }
        if imageView.layer.cornerRadius != 0 {
            imageView.layer.cornerRadius = 0
        }
    }

    func setImage(_ image: UIImage?) {
        if imageView.image !== image {
            imageView.image = image
        }
    }
}
