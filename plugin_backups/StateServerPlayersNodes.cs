// File: StateServerPlayersNodes.cs
// Drop into: <your_server>/oxide/plugins/
// Listens on 127.0.0.1:5555 and responds to GET_STATE with a JSON snapshot of players + ore nodes.
//
// Notes:
//  - Uses timer.NextFrame(...) to ensure all Unity/Rust object access is main-thread safe.
//  - Background TCP listener is used for socket I/O so Oxide's main thread isn't blocked.
//  - Includes a small placeholder branch for future commands (CMD|...).
//  - Keep BuildSnapshot() as the primary place to modify returned JSON fields.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Oxide.Core;
using Oxide.Core.Plugins;
using UnityEngine;

/// <summary>
/// Oxide plugin that provides a local TCP server for retrieving real-time game state data.
/// Listens on 127.0.0.1:5555 and returns JSON snapshots of active players and ore nodes.
/// </summary>
/// <remarks>
/// Protocol:
/// - Client connects and sends "GET_STATE" to receive a JSON snapshot containing:
///   - Player data: position, rotation, health, metabolism, inventory items
///   - Ore node data: position, health, prefab name (limited to MAX_NODES_RETURNED)
/// - Client sends "CMD|..." for future command support (currently returns not_implemented)
/// - All responses are newline-terminated
/// 
/// Error responses:
/// - "ERR|empty_request" - No command received
/// - "ERR|timeout" - Snapshot generation exceeded SNAPSHOT_TIMEOUT_MS
/// - "ERR|snapshot_error:{message}" - Exception during snapshot creation
/// - "ERR|cmd_not_implemented" - Command protocol not yet implemented
/// - "ERR|unsupported_command" - Unknown command received
/// 
/// Thread safety:
/// - TCP listener and client handling run on background threads
/// - Snapshot generation (BuildSnapshot) runs on main thread via timer.NextFrame
/// - All game object access is confined to main thread to avoid threading issues
/// 
/// Configuration constants:
/// - PORT: 5555 (localhost only)
/// - SOCKET_TIMEOUT_MS: 5000 (read/write timeout)
/// - SNAPSHOT_TIMEOUT_MS: 4000 (maximum time to wait for snapshot)
/// - MAX_NODES_RETURNED: 1000 (prevents oversized payloads on dense maps)
/// </remarks>
namespace Oxide.Plugins
{
    [Info("StateServerPlayersNodes", "You", "0.2")]
    [Description("Local state server: returns players and ore node data via TCP on 127.0.0.1:5555")]
    public class StateServerPlayersNodes : CovalencePlugin
    {
        private TcpListener listener;
        private volatile bool running = false;
        private const int PORT = 5555;
        private const int SOCKET_TIMEOUT_MS = 5000;
        private const int SNAPSHOT_TIMEOUT_MS = 4000;
        private const int READ_BUFFER = 16384;

        #region Oxide lifecycle
        void OnServerInitialized()
        {
            StartListener();
            Puts($"StateServerPlayersNodes loaded - listening on 127.0.0.1:{PORT}");
        }

        void Unload()
        {
            StopListener();
            Puts("StateServerPlayersNodes unloaded");
        }
        #endregion

        #region Listener lifecycle
        private void StartListener()
        {
            if (running) return;
            running = true;
            Task.Run(async () =>
            {
                try
                {
                    listener = new TcpListener(IPAddress.Loopback, PORT);
                    listener.Start();
                }
                catch (Exception ex)
                {
                    PrintError($"StateServer: failed to start listener: {ex}");
                    running = false;
                    return;
                }

                while (running)
                {
                    TcpClient client = null;
                    try
                    {
                        client = await listener.AcceptTcpClientAsync();
                        _ = HandleClientAsync(client); // fire-and-forget
                    }
                    catch (ObjectDisposedException)
                    {
                        // listener stopped
                        break;
                    }
                    catch (Exception ex)
                    {
                        PrintWarning($"StateServer: Accept error: {ex}");
                        client?.Close();
                        await Task.Delay(100);
                    }
                }
            });
        }

        private void StopListener()
        {
            try
            {
                running = false;
                if (listener != null)
                {
                    listener.Stop();
                    listener = null;
                }
            }
            catch (Exception ex)
            {
                PrintWarning($"StateServer: stop error: {ex}");
            }
        }
        #endregion

        #region Client handling & protocol
        private async Task HandleClientAsync(TcpClient client)
        {
            using (client)
            {
                client.NoDelay = true;
                client.ReceiveTimeout = SOCKET_TIMEOUT_MS;
                client.SendTimeout = SOCKET_TIMEOUT_MS;

                NetworkStream stream = null;
                try
                {
                    stream = client.GetStream();
                }
                catch
                {
                    return;
                }

                var buffer = new byte[READ_BUFFER];
                int read = 0;
                try
                {
                    read = await stream.ReadAsync(buffer, 0, buffer.Length);
                }
                catch
                {
                    return;
                }

                if (read <= 0) return;

                var req = Encoding.UTF8.GetString(buffer, 0, read).Trim();
                // Normalize to first-line only in case client sends trailing data
                var firstLine = req.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).FirstOrDefault()?.Trim();
                if (string.IsNullOrEmpty(firstLine))
                {
                    await WriteStringAsync(stream, "ERR|empty_request\n");
                    return;
                }

                // Command routing: GET_STATE is primary; add future commands under CMD|...
                if (firstLine.Equals("GET_STATE", StringComparison.OrdinalIgnoreCase))
                {
                    var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

                    // Schedule main-thread snapshot
                    NextFrame(() =>
                    {
                        try
                        {
                            var helper = plugins.Find("GetState") as GetState;
                            if (helper == null)
                            {
                                tcs.TrySetException(new Exception("GetState helper not loaded"));
                            }
                            else
                            {
                                var snapshot = helper.BuildSnapshot(); // runs on main thread
                                var json = JsonConvert.SerializeObject(snapshot);
                                tcs.TrySetResult(json);
                            }
                        }
                        catch (Exception ex)
                        {
                            tcs.TrySetException(ex);
                        }
                    });

                    // Await snapshot or timeout
                    var winner = await Task.WhenAny(tcs.Task, Task.Delay(SNAPSHOT_TIMEOUT_MS));
                    if (winner != tcs.Task)
                    {
                        await WriteStringAsync(stream, "ERR|timeout\n");
                        return;
                    }

                    string respJson;
                    try
                    {
                        respJson = tcs.Task.Result + "\n";
                    }
                    catch (Exception ex)
                    {
                        await WriteStringAsync(stream, $"ERR|snapshot_error:{ex.Message}\n");
                        return;
                    }

                    await WriteBytesAsync(stream, Encoding.UTF8.GetBytes(respJson));
                }
                else if (firstLine.Equals("RESET_ENV", StringComparison.OrdinalIgnoreCase))
                {
                    // Handle RESET_ENV command
                    var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

                    // Schedule main-thread execution
                    NextFrame(() =>
                    {
                        try
                        {
                            var helper = plugins.Find("ResetEnv") as ResetEnv;
                            if (helper == null)
                            {
                                tcs.TrySetResult("ERR|training_helper_not_loaded");
                            }
                            else
                            {
                                var result = helper.ResetTrainingEnvironment();
                                tcs.TrySetResult(result);
                            }
                        }
                        catch (Exception ex)
                        {
                            tcs.TrySetException(ex);
                        }
                    });

                    // Await result or timeout
                    var winner = await Task.WhenAny(tcs.Task, Task.Delay(SNAPSHOT_TIMEOUT_MS));
                    if (winner != tcs.Task)
                    {
                        await WriteStringAsync(stream, "ERR|timeout\n");
                        return;
                    }

                    string result;
                    try
                    {
                        result = tcs.Task.Result + "\n";
                    }
                    catch (Exception ex)
                    {
                        await WriteStringAsync(stream, $"ERR|command_error:{ex.Message}\n");
                        return;
                    }

                    await WriteStringAsync(stream, result);
                }
                else if (firstLine.Equals("GIVE_ITEM", StringComparison.OrdinalIgnoreCase))
                {
                    // Handle GIVE_ITEM command
                    var tcs = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);

                    // Schedule main-thread execution
                    NextFrame(() =>
                    {
                        try
                        {
                            var helper = plugins.Find("GiveItem") as GiveItem;
                            if (helper == null)
                            {
                                tcs.TrySetResult("ERR|giveitem_helper_not_loaded");
                            }
                            else
                            {
                                var result = helper.GivePickaxe();
                                tcs.TrySetResult(result);
                            }
                        }
                        catch (Exception ex)
                        {
                            tcs.TrySetException(ex);
                        }
                    });

                    // Await result or timeout
                    var winner = await Task.WhenAny(tcs.Task, Task.Delay(SNAPSHOT_TIMEOUT_MS));
                    if (winner != tcs.Task)
                    {
                        await WriteStringAsync(stream, "ERR|timeout\n");
                        return;
                    }

                    string result;
                    try
                    {
                        result = tcs.Task.Result + "\n";
                    }
                    catch (Exception ex)
                    {
                        await WriteStringAsync(stream, $"ERR|command_error:{ex.Message}\n");
                        return;
                    }

                    await WriteStringAsync(stream, result);
                }
                else
                {
                    await WriteStringAsync(stream, "ERR|unsupported_command\n");
                }
            }
        }

        private Task WriteStringAsync(NetworkStream stream, string s)
        {
            return WriteBytesAsync(stream, Encoding.UTF8.GetBytes(s));
        }

        private async Task WriteBytesAsync(NetworkStream stream, byte[] bytes)
        {
            try
            {
                await stream.WriteAsync(bytes, 0, bytes.Length);
            }
            catch
            {
                // ignore write errors (client closed)
            }
        }
        #endregion
    }
}
