import SwiftUI
import UIKit

struct AvatarFrameView: UIViewRepresentable {
    let image: UIImage?
    var cornerRadius: CGFloat = 20

    func makeUIView(context: Context) -> UIImageView {
        let imageView = UIImageView()
        imageView.backgroundColor = .clear
        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = cornerRadius
        imageView.layer.cornerCurve = .continuous
        return imageView
    }

    func updateUIView(_ uiView: UIImageView, context: Context) {
        if uiView.image !== image {
            uiView.image = image
        }
        if uiView.layer.cornerRadius != cornerRadius {
            uiView.layer.cornerRadius = cornerRadius
        }
    }
}
