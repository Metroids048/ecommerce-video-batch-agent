// Native Vision rectangle proposals only; no network or model download.
import Foundation
import Vision
import AVFoundation
let args=CommandLine.arguments
if args.count != 3 { exit(2) }
guard let stride=Int(args[2]),stride>0 else { exit(2) }
let asset=AVURLAsset(url:URL(fileURLWithPath:args[1]))
guard let track=asset.tracks(withMediaType:.video).first else { exit(2) }
let reader=try AVAssetReader(asset:asset)
let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
reader.add(output)
func emit(_ row:[String:Any]) {
 let data=try! JSONSerialization.data(withJSONObject:row,options:[.sortedKeys])
 print(String(data:data,encoding:.utf8)!);fflush(stdout)
}
reader.startReading()
var n=0,scanned=0
while let sample=output.copyNextSampleBuffer() {
 defer { n += 1 }
 if n % stride != 0 { continue }
 try autoreleasepool {
  guard let pixels=CMSampleBufferGetImageBuffer(sample) else { exit(2) }
  let request=VNDetectRectanglesRequest()
  request.maximumObservations=50;request.minimumConfidence=0.2
  request.minimumAspectRatio=0.15;request.maximumAspectRatio=1.0
  request.minimumSize=0.025;request.quadratureTolerance=35
  try VNImageRequestHandler(cvPixelBuffer:pixels,orientation:.up,options:[:]).perform([request])
  let rows=(request.results ?? []).map { o -> [String:Any] in
   ["confidence":o.confidence,"quad":[o.topLeft,o.topRight,o.bottomRight,o.bottomLeft].map {[$0.x,$0.y]}]
  }
  emit(["frame":n,"width":CVPixelBufferGetWidth(pixels),"height":CVPixelBufferGetHeight(pixels),"rectangles":rows]);scanned += 1
 }
}
if reader.status != .completed { exit(1) }
emit(["complete":true,"decoded_frames":n,"carrier_frames":scanned,"stride":stride])
