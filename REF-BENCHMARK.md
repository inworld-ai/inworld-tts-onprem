# Latency Benchmarks for tts-1.5-mini-onprem

## First Chunk Streaming Latency P50 and P90 in ms

Every chunk is 0.9 s audio play time

| Target QPS | Avg QPS (B200/H100) | P50 (B200/H100) | P50 improv (B200 vs H100) | P90 (B200/H100) | P90 improv (B200 vs H100) |
| ---------: | :------------------ | :-------------- | :------------------------ | :-------------- | :------------------------ |
|          1 | 0.94 / 0.97         | 169.5 / 202.5   | 16.3%                     | 232.2 / 271.2   | 14.4%                     |
|          3 | 2.91 / 3.19         | 207.3 / 249.0   | 16.7%                     | 263.8 / 312.8   | 15.7%                     |
|          5 | 5.42 / 5.24         | 230.5 / 284.2   | 18.9%                     | 278.4 / 371.0   | 25.0%                     |
|          7 | 7.64 / 6.64         | 257.4 / 323.5   | 20.4%                     | 336.9 / 414.4   | 18.7%                     |

## E2E Generation Latency P50 and P90 in ms

| Target QPS | Avg QPS (B200/H100) | P50 (B200/H100) | P50 improv (B200 vs H100) | P90 (B200/H100) | P90 improv (B200 vs H100) |
| ---------: | :------------------ | :-------------- | :------------------------ | :-------------- | :------------------------ |
|          1 | 0.94 / 0.97         | 1442.3 / 1769.5 | 18.5%                     | 1606.3 / 1962.0 | 18.1%                     |
|          3 | 2.91 / 3.19         | 1600.0 / 2007.4 | 20.3%                     | 1809.2 / 2195.8 | 17.6%                     |
|          5 | 5.42 / 5.24         | 1737.4 / 2253.5 | 22.9%                     | 1946.4 / 2541.0 | 23.4%                     |
|          7 | 7.64 / 6.64         | 1942.6 / 2460.6 | 21.0%                     | 2196.1 / 2848.3 | 22.9%                     |

## Supported Concurrent Sessions

| Target QPS | Avg QPS (B200/H100) | P50 (B200/H100) | P50 improv (B200 vs H100) | P90 (B200/H100) | P90 improv (B200 vs H100) |
| ---------: | :------------------ | :-------------- | :------------------------ | :-------------- | :------------------------ |
|          1 | 0.94 / 0.97         | 1.4 / 1.7       | 21.1%                     | 1.5 / 1.9       | 20.7%                     |
|          3 | 2.91 / 3.19         | 4.7 / 6.4       | 27.1%                     | 5.3 / 7.0       | 24.7%                     |
|          5 | 5.42 / 5.24         | 9.4 / 11.8      | 20.1%                     | 10.6 / 13.3     | 20.6%                     |
|          7 | 7.64 / 6.64         | 14.8 / 16.3     | 9.2%                      | 16.8 / 18.9     | 11.4%                     |

* Overall, the B200 delivers \~20% lower first-chunk latency (an improvement on the order of tens of milliseconds). Once streaming begins and the first chunk is received, any residual differences in subsequent chunk latency are effectively masked and are unlikely to be perceptible to end users.  
* The H100, however, can sustain a higher number of concurrent sessions. Because its end-to-end (E2E) latency is longer, requests remain in flight for a longer duration on average—resulting in higher observed concurrency at a given throughput.  
* **Note:** These results reflect a V0 configuration without full optimizations. If needed, we can further reduce first-chunk latency with additional tuning.

## Benchmark dataset and Hardware Specifications

| Model | Hardware | Os  | Cuda version | CPU | Dataset | Load test setting | Inference Stack and Env |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| TTS 1.5 1B | 1 H100 SXM5 node | Ubuntu 22.04.5 LTS | CUDA 13.0 driver version 580.95.05 with kernel driver version 570.195.03 | CPU: Intel® Xeon® Platinum 8480+ Architecture:   • CPU family: 6   • Model: 143 Topology:   • Cores per socket: 13   • Threads per core: 2  | Single fixed length request (30 words utterance) | Based on the QPS setting, the client sends the same requests to the server. Measure chunk and E2E latency at client side.  | TRTLLM+PytorchDocker image: [**nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4**](http://nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4) |
| TTS 1.5 1B | 1 B100 SXM6 node | Ubuntu 22.04.5 LTS | CUDA 13.0 driver version 580.95.05 with kernel driver version 570.195.03 | CPU: AMD EPYC™ 7J13 64-Core Processor Architecture:   • CPU family: 25   • Model: 1 Topology:   • Cores per socket: 64   • Threads per core: 1  | Single fixed length request (30 words utterance) | Based on the QPS setting, the client sends the same requests to the server. Measure chunk and E2E latency at client side.  | TRTLLM+PytorchDocker image: [**nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4**](http://nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc4) |
