package com.example.gabai.services

import android.content.Context
import android.net.wifi.WifiManager
import org.json.JSONObject
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

/**
 * Finds the relay on the local hotspot with a UDP query (see gabai/relay/discovery.py). No internet, no multicast.
 * The query goes to the limited broadcast, the DHCP broadcast and the gateway: the Windows hotspot gateway is the PC itself.
 */
object RelayDiscovery {
    private const val PORT = 8091
    private val QUERY = "GABAI?".toByteArray()

    /** Blocking, up to [timeoutMs]. Returns "http://ip:port" or null. Call off the main thread. */
    fun find(c: Context, timeoutMs: Int = 2500): String? = runCatching {
        DatagramSocket().use { s ->
            s.broadcast = true
            s.soTimeout = timeoutMs
            targets(c).forEach { ip -> runCatching { s.send(DatagramPacket(QUERY, QUERY.size, ip, PORT)) } }
            val buf = ByteArray(256)
            val p = DatagramPacket(buf, buf.size)
            s.receive(p)
            val port = JSONObject(String(p.data, 0, p.length)).getInt("relay_port")
            "http://${p.address.hostAddress}:$port"
        }
    }.getOrNull()

    /** Finds the relay and, when found, makes it the saved address. Returns true if the saved address was set. */
    fun refresh(c: Context): Boolean {
        val url = find(c) ?: gatewayRelay(c) ?: scanSubnet(c) ?: return false
        SettingsStore.setRelayUrl(c, url)
        return true
    }

    /** The relay's address if [host] answers /api/health like a GabAi relay, else null. */
    private fun probe(host: String, timeoutMs: Int = 1500): String? = runCatching {
        val url = "http://$host:8090"
        val conn = java.net.URL("$url/api/health").openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = timeoutMs; conn.readTimeout = timeoutMs
        val ok = conn.responseCode == 200 && JSONObject(conn.inputStream.bufferedReader().readText()).optBoolean("ok")
        conn.disconnect()
        if (ok) url else null
    }.getOrNull()

    private fun dotted(v: Int) = "${v and 0xFF}.${(v shr 8) and 0xFF}.${(v shr 16) and 0xFF}.${(v shr 24) and 0xFF}"

    /** UDP can be blocked by the PC firewall while TCP 8090 is open: on the PC's own hotspot the gateway is the PC, so ask it over HTTP. */
    @Suppress("DEPRECATION")
    private fun gatewayRelay(c: Context): String? = runCatching {
        val gw = (c.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager).dhcpInfo.gateway
        if (gw != 0) probe(dotted(gw)) else null
    }.getOrNull()

    /** Last resort, for a PC on a router's Wi-Fi: ask every address of this phone's /24 (about 2 s on a home network). */
    @Suppress("DEPRECATION")
    private fun scanSubnet(c: Context): String? = runCatching {
        val me = (c.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager).dhcpInfo.ipAddress
        if (me == 0) return@runCatching null
        val prefix = dotted(me).substringBeforeLast('.')
        val pool = java.util.concurrent.Executors.newFixedThreadPool(48)
        try {
            pool.invokeAny((1..254).map { i -> java.util.concurrent.Callable { probe("$prefix.$i", 600) ?: throw IllegalStateException() } }, 8, java.util.concurrent.TimeUnit.SECONDS)
        } finally { pool.shutdownNow() }
    }.getOrNull()

    @Suppress("DEPRECATION")
    private fun targets(c: Context): List<InetAddress> {
        val out = mutableListOf(InetAddress.getByName("255.255.255.255"))
        val dhcp = (c.applicationContext.getSystemService(Context.WIFI_SERVICE) as? WifiManager)?.dhcpInfo ?: return out
        fun ip(v: Int) = InetAddress.getByAddress(byteArrayOf(v.toByte(), (v shr 8).toByte(), (v shr 16).toByte(), (v shr 24).toByte()))
        if (dhcp.gateway != 0) out += ip(dhcp.gateway)
        if (dhcp.ipAddress != 0 && dhcp.netmask != 0) out += ip(dhcp.ipAddress or dhcp.netmask.inv())
        return out
    }
}
