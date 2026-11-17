// Renders an SVG centred on a transparent square PNG (AppKit draws the SVG, CSS styles and
// gradients included). Usage: swift scripts/render_svg.swift <in.svg> <out.png> <pixels> <margin>
// where margin is the share of the side left empty on each edge.
import AppKit

let args = CommandLine.arguments
guard args.count == 5, let size = Int(args[3]), let margin = Double(args[4]) else {
    fatalError("usage: render_svg.swift <in.svg> <out.png> <pixels> <margin>")
}
guard let image = NSImage(contentsOf: URL(fileURLWithPath: args[1])) else {
    fatalError("cannot read \(args[1])")
}
let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8, samplesPerPixel: 4,
    hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
NSGraphicsContext.current?.imageInterpolation = .high
let side = Double(size) * (1 - 2 * margin)
let scale = min(side / image.size.width, side / image.size.height)
let (w, h) = (image.size.width * scale, image.size.height * scale)
image.draw(in: NSRect(x: (Double(size) - w) / 2, y: (Double(size) - h) / 2, width: w, height: h))
NSGraphicsContext.restoreGraphicsState()
try rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[2]))
