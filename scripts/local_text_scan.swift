// On-device macOS Vision only. No service, model download or network client.
import Foundation
import Vision
import AVFoundation
import CoreImage
import ImageIO

let args=CommandLine.arguments
if args.count != 4 { fputs("source, custom words, stride required\n",stderr); exit(2) }
let stride=Int(args[3])!
let asset=AVURLAsset(url:URL(fileURLWithPath:args[1]))
guard let track=asset.tracks(withMediaType:.video).first else { exit(2) }
let reader=try AVAssetReader(asset:asset)
let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
reader.add(output)
let context=CIContext(options:[.cacheIntermediates:false])
let orientations:[(String,CGImagePropertyOrientation)]=[("up",.up),("rightMirrored",.rightMirrored)]
func emit(_ value:[String:Any]) {
 let data=try! JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])
 print(String(data:data,encoding:.utf8)!);fflush(stdout)
}
reader.startReading()
var n=0
var scanned=0
let start=Date()
while let sample=output.copyNextSampleBuffer() {
 defer { n += 1 }
 if n % stride != 0 { continue }
 try autoreleasepool {
  guard let pixels=CMSampleBufferGetImageBuffer(sample) else { throw NSError(domain:"pixels",code:1) }
  let width=CVPixelBufferGetWidth(pixels),height=CVPixelBufferGetHeight(pixels)
  let scale=min(1,720.0/Double(width))
  let image=CIImage(cvPixelBuffer:pixels).transformed(by:CGAffineTransform(scaleX:scale,y:scale))
  guard let cg=context.createCGImage(image,from:image.extent) else { throw NSError(domain:"image",code:1) }
  var passes:[[String:Any]]=[]
  for (name,orientation) in orientations {
   let req=VNRecognizeTextRequest()
   req.recognitionLevel = .accurate
   req.usesLanguageCorrection = false
   req.recognitionLanguages = ["en-US","fr-FR"]
   req.customWords = args[2].components(separatedBy:"|")
   req.minimumTextHeight = 0.006
   try VNImageRequestHandler(cgImage:cg,orientation:orientation,options:[:]).perform([req])
   let rows=(req.results ?? []).compactMap { o -> [String:Any]? in
    guard let c=o.topCandidates(1).first else { return nil }
    return ["text":c.string,"confidence":c.confidence,"quad":[o.topLeft,o.topRight,o.bottomRight,o.bottomLeft].map { [$0.x,$0.y] }]
   }
   passes.append(["orientation":name,"rows":rows])
  }
  emit(["frame":n,"width":width,"height":height,"passes":passes])
  scanned += 1
  if scanned % 100 == 0 { fputs("OCR \(scanned) frames, \(Int(Date().timeIntervalSince(start)))s\n",stderr) }
 }
}
if reader.status == .failed { fputs("source decode failed: \(String(describing:reader.error))\n",stderr);exit(1) }
emit(["complete":true,"decoded_frames":n,"ocr_frames":scanned,"stride":stride,"seconds":Date().timeIntervalSince(start)])
