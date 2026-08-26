from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel

def create_topology():
    net = Mininet(controller=Controller)
    
    print("*** Adding controller")
    net.addController('c0')
    h1 = net.addHost('h1')
    h2 = net.addHost('h2')
    s1 = net.addSwitch('s1')
    net.addLink(h1, s1)
    net.addLink(h2, s1)

    
    net.start()
    print("*** Starting packet capture on h2")
    h2.cmd('tcpdump -i h2-eth0 -w /home/dark/Documents/q-ids/capture/raw/test_capture.pcap &')
    print("*** Running CLI")
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_topology()