# microduck-brain

Creier autonom **v0** pentru Microduck: un program separat, client al robotului, care vorbește cu el
exact ca `robotctl` (JSON-RPC 2.0, un obiect pe linie, pe socket-uri unix). Nu modifică și nu
depinde de repository-urile Pollen. Python 3.12, `uv`, **zero dependențe** în afară de biblioteca
standard.

Ce face:

1. merge înainte cu viteză mică (`robot.move`, 20 Hz);
2. citește cadrele de adâncime 8×8; când are un obstacol aproape în față, se oprește, se întoarce
   spre partea mai liberă și continuă;
3. dacă face mai multe ocoliri la rând fără să apuce să meargă înainte (colț), face o întoarcere mare;
4. dacă robotul raportează că a căzut sau că nu e sănătos, **nu mai trimite comenzi de mers** până e
   din nou în picioare (și mai așteaptă `resume_delay`);
5. la Ctrl+C (sau SIGTERM) trimite `robot.stop` și iese curat, cu un rezumat.

## Pornire pe PC (WSL Ubuntu)

Presupun repository-urile în `~/microduck` și `~/microduck_rl`, iar acest repository în
`~/microduck-brain`.

```bash
# o singură dată: uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# terminalul 1: simularea (scena apartment, fără fereastră — vezi nota de mai jos)
DUCK_SIM_SCENE=apartment DUCK_SIM_VIEWER=0 bash ~/microduck_sim_up.sh

# terminalul 2: creierul
cd ~/microduck-brain
uv run microduck-brain --config brain.toml
```

Oprești cu **Ctrl+C**. Pentru o rulare limitată: `uv run microduck-brain --duration 180`.

Teste: `uv run pytest`.

**Socket-urile** se găsesc singure: dacă există `~/.cache/duck-sim/duck-a.sock` și
`duck-a-tof.sock` (locul unde le pune `scripts/duck-sim`), le folosește pe acelea; altfel
`/run/robotd.sock` și `/run/tofd/tof.sock` (robotul real). Dacă scriptul tău de pornire folosește alt
director, setează `DUCK_SIM_STATE=/calea/mea`, sau dă căile direct:
`--robot-socket ... --tof-socket ...` (sau în `[sockets]` din config).

> Nu știu ce face `~/microduck_sim_up.sh` — nu l-am văzut. Dacă el nu respectă variabilele
> `DUCK_SIM_*` ale lui `scripts/duck-sim`, pornește simularea cum o pornești de obicei și
> folosește `--robot-socket` / `--tof-socket`.

## Parametri (`brain.toml`)

Toate valorile din fișier sunt cele implicite; poți șterge orice linie. O cheie scrisă greșit dă
eroare la pornire, nu e ignorată în tăcere. Distanțe în metri, viteze în m/s sau rad/s, timpuri în
secunde.

### `[sockets]`
| Cheie | Ce face |
|---|---|
| `robot` | socket-ul lui `robotd`. Gol = se ghicește (vezi mai sus). |
| `tof` | socket-ul lui `tofd` (adâncimea). Gol = se ghicește. |

### `[motion]`
| Cheie | Ce face |
|---|---|
| `forward_speed` | viteza de mers înainte. Robotul acceptă până la ~0,3 m/s; 0,10 e „încet”. |
| `turn_speed` | cât de repede se întoarce când ocolește (rad/s; pozitiv = stânga). |
| `big_turn_speed` | cât de repede se întoarce la întoarcerea mare. |

### `[obstacle]`
| Cheie | Ce face |
|---|---|
| `stop_distance` | un obstacol mai aproape de atât în față => se oprește. |
| `clear_distance` | ca să termine întoarcerea trebuie să vadă liber cel puțin atât (mai mare decât `stop_distance`, ca să nu oscileze). |
| `first_row`, `last_row` | rândurile 8×8 folosite (0 = sus). Cele de jos pot vedea podeaua; în simulare, în picioare, podeaua apărea doar pe marginile rândurilor 5–7, deci 0–4 e sigur. |
| `front_first_col`, `front_last_col` | coloanele care contează „în față” (0 = stânga). Implicit 1–6: marginile (0 și 7) nu opresc robotul. |
| `min_zones` | câte zone trebuie să fie aproape ca să conteze ca obstacol. 2 taie zgomotul de o singură zonă. |
| `blind_fraction` | dacă sub atâta din zonele din față dau o măsurătoare validă (senzor orb, perete lipit de el), se tratează ca obstacol. |
| `max_range` | „nimic în rază” (status 255) se socotește la această distanță. |
| `stop_pause` | cât stă pe loc după oprire, înainte să înceapă să se întoarcă. |
| `min_turn_time` | întoarcerea durează cel puțin atât. |
| `max_turn_time` | dacă nu se eliberează în atâta timp, renunță și face întoarcerea mare. |
| `default_turn` | `left` sau `right`: direcția când ambele părți sunt la fel de libere. |

### `[corner]`
| Cheie | Ce face |
|---|---|
| `turns_before_big_turn` | atâtea ocoliri la rând, fără progres între ele => întoarcere mare. |
| `progress_time` | cât trebuie să meargă înainte nestingherit ca numărătoarea de ocoliri să revină la zero. |
| `big_turn_degrees` | unghiul întoarcerii mari. E **cronometrat** (unghi / `big_turn_speed`), nu măsurat; unghiul real depinde de cât de bine urmează robotul comanda. |

### `[safety]`
| Cheie | Ce face |
|---|---|
| `tof_timeout` | un cadru de adâncime mai vechi de atât => robotul e oprit (nu merge orb). |
| `state_timeout` | dacă fluxul `robot.state` tace atât, robotul e considerat nesigur. |
| `health_poll` | cât de des întreabă `robot.health`. |
| `resume_delay` | după ce robotul e iar sigur (în picioare, sănătos, cu adâncime), mai așteaptă atât, continuu, înainte să meargă. |

### `[loop]`
| Cheie | Ce face |
|---|---|
| `rate_hz` | cât de des trimite `robot.move`. Robotul oprește singur mersul dacă nu primește comenzi ~0,5 s (deadman), deci nu coborî sub ~5 Hz. |
| `log_every` | la câte secunde scrie o linie de stare (0 = deloc). |

## Cum e făcut

```
src/microduck_brain/
  config.py   configurația (dataclass-uri + TOML, chei necunoscute respinse)
  depth.py    cadru 8×8 -> „blocat?”, „care parte e mai liberă?”
  brain.py    mașina de stări; logică pură, fără socket-uri și fără ceas propriu
  ipc.py      JSON-RPC pe socket unix (cereri, notificări, fluxuri abonate)
  runner.py   buclă: citește senzorii, apelează Brain.tick, trimite comenzi
  cli.py      linia de comandă
```

Stări: `hold → forward → brake → turn → forward`, cu `bigturn` pentru colțuri. Un „gardian” din
`Brain.tick` stă deasupra tuturor stărilor: căzut / nesănătos / fără adâncime => `hold`.

**Stare nouă** (dans, odihnă): o clasă cu `name` și `step(brain, inputs) -> (Command, next | None)`,
înregistrată cu `brain.register(...)`. Dacă nu are nevoie de adâncime, `needs_depth = False`.
Detalii în docstring-ul din `brain.py`.

## Limite cunoscute (v0)

* Dacă robotul a căzut, creierul **doar așteaptă**; nu încearcă să-l ridice.
* Întoarcerea mare e cronometrată, nu urmărește odometria.
* Nu vede obstacole sub nivelul rândurilor folosite (praguri, obiecte joase) și nici sticlă.
* Orientarea senzorului (coloana 0 = stânga) e luată din simulator și din `tofd`; pe robotul real
  merită verificată.
