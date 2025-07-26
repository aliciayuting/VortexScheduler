"""      --------        Network Transfer delay Parameters        -------- 
# Delta of transfer delay : in the unit of millisecond
# message_size in the unit of kiloByte
# RDMA has different bandwidth, here we are using 100 Gb/s ~ 12.5 GB/s ~ 12.5 e+6 KB/s
"""
FIXED_COST_GROUP_FORMATION = 0.002  # in ms
RDMA_BANDWIDTH = 12.5 * (1 << 20)


# CPU->CPU:  CPU to CPU RDMA delay, since read and write have similar throughput
# here we approximate them to estimate CPUtoCPU delay
def CPU_to_CPU_delay(message_size) -> float:
    """
    Message size in kB
    Return the delay in ms
    """

    # if message_size < 1:
    #     return 2E-3  # 2 microseconds
    # elif message_size < (1 << 2):  # messages of size between 1 and 4 kB
    #     through_put = (3 + message_size * 2) * (1 << 20) / 1000
    #     return (message_size / through_put) + FIXED_COST_GROUP_FORMATION
    # else:
    #     through_put = 12 * 1048.58 # (kB/ms) Using the numbers from Cascade's benchmark
    #     return (message_size / through_put) + FIXED_COST_GROUP_FORMATION

    return message_size / 250000


def SameMachineCPUtoGPU_delay(message_size) -> float:
    """
    message_size expressed in kB
    Returns delay in ms
    """

    pcie_bw = 64000
    # through_put = 9.7 * 1048.58  # (kB/ms) Using the numbers from Cascade's benchmark
    return (message_size / pcie_bw) 


def UplinkEdgeToCloud_delay(message_size) -> float:
    """
    message_size expressed in kB
    Returns delay in ms
    https://arxiv.org/pdf/2109.03395.pdf
    LTE: mean 14 Mbps
    WiFi: mean 26 Mbps -> mean 3.25 MBps
    5G: mean 76 Mbps
    """
    # Using WiFi
    through_put = (26 / 8) * 1024 / 1000.0  # (kB/ms)
    return message_size / through_put


def DownlinkCloudToEdge_delay(message_size) -> float:
    """
    message_size expressed in kB
    Returns delay in ms
    https://arxiv.org/pdf/2109.03395.pdf
    LTE: mean 42 Mbps
    WiFi: mean 41 Mbps
    5G: mean 497 Mbps
    """
    # Using WiFi
    through_put = (41 / 8) * 1024 / 1000  # (kB/ms)
    return message_size / through_put


def SameMachineGPUtoCPU_delay(message_size):
    """
    message_size expressed in kB
    Returns delay in ms
    """
    pcie_bw = 64000
    # through_put = 9.7 * 1048.58   # (kB/ms) Using the numbers from Cascade's benchmark
    return (message_size / pcie_bw) 


def GPU_to_GPU_delay(message_size) -> float:
    """
    GPU->GPU: local GPU -> local CPU -> remote CPU* -> remote GPU*
    """

    delay_localGPU_to_localCPU = SameMachineGPUtoCPU_delay(message_size)
    delay_localCPU_to_remoteCPU = CPU_to_CPU_delay(message_size)
    delay_remoteCPU_to_remoteGPU = SameMachineCPUtoGPU_delay(message_size)
    # print("message size", message_size, "GPU TO GPU DELAY: ", delay_localGPU_to_localCPU + delay_localCPU_to_remoteCPU + delay_remoteCPU_to_remoteGPU)
    return delay_localGPU_to_localCPU + delay_localCPU_to_remoteCPU + delay_remoteCPU_to_remoteGPU
