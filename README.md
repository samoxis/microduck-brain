# microduck-brain

Creier autonom **v0** pentru Microduck: un program separat, client al robotului, care vorbește cu el
exact ca `robotctl` (JSON-RPC 2.0, un obiect pe linie, pe socket-uri unix). Nu modifică și nu
depinde de repository-urile Pollen. Python 3.12, `uv`, **zero dependențe** în afară de biblioteca
standard.

Ce face:

1. merge înainte (`robot.move`, 20 Hz) și **ține direcția** cu `odom.yaw` (corectează deriva);
2. citește cadrele de adâncime 8×8; când un obstacol se apropie, **virează din mers** spre partea mai
   liberă, mai tare cu cât e mai aproape; se oprește de tot și se rotește pe loc doar când nu mai are
   loc (sub `stop_distance`);
3. dacă ajunge de mai multe ori la frână fără să apuce să meargă înainte (colț), face o întoarcere
   mare; **rotirile se măsoară din `odom.yaw`**, nu din timp;
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
Vitezele sunt cele **cerute**; cele reale ies mai mici (măsurat: 0,24 cerut ≈ 0,10 m/s real).
Praguri măsurate în simulare, scrise în cod (`MIN_FORWARD_SPEED = 0,20 m/s`, `MIN_TURN_SPEED = 1,1 rad/s`):
sub ele politica nu pășește deloc. Dacă configurația cere mai puțin, la pornire apare un avertisment
`ATENȚIE` pe fiecare cheie în parte (creierul pornește oricum).

| Cheie | Ce face |
|---|---|
| `forward_speed` | viteza de mers înainte cerută (m/s). Minim 0,20. |
| `turn_speed` | cât de repede se rotește în loc când nu mai are loc (rad/s; pozitiv = stânga). Minim 1,1. |
| `big_turn_speed` | cât de repede se rotește la întoarcerea mare. Minim 1,1. |

### `[obstacle]`
| Cheie | Ce face |
|---|---|
| `stop_distance` | un obstacol mai aproape de atât în față (nu mai are loc) => se oprește de tot și se rotește pe loc. |
| `slow_distance` | între `slow_distance` și `stop_distance` virează din mers (fără să se oprească), cu atât mai tare cu cât obstacolul e mai aproape. Direcția se alege la începutul ocolirii (partea mai liberă) și nu se schimbă cât durează. |
| `steer_yaw_max` | rotirea cerută (rad/s) a virajului din mers, când obstacolul e la `stop_distance`. Nu are prag minim cunoscut: când mergi și virezi simultan, dacă robotul nu pare să vireze, ridic-o. |
| `steer_slew` | cât de repede poate varia rotirea cerută (rad/s pe secundă): fără smucituri. |
| `clear_distance` | ca să termine întoarcerea trebuie să vadă liber cel puțin atât (mai mare decât `stop_distance`, ca să nu oscileze). |
| `first_row`, `last_row` | rândurile 8×8 folosite (0 = sus). Cele de jos pot vedea podeaua; în simulare, în picioare, podeaua apărea doar pe marginile rândurilor 5–7, deci 0–4 e sigur. |
| `front_first_col`, `front_last_col` | coloanele care contează „în față” (0 = stânga). Implicit 1–6: marginile (0 și 7) nu opresc robotul. |
| `min_zones` | câte zone trebuie să fie aproape ca să conteze ca obstacol. 2 taie zgomotul de o singură zonă. |
| `blind_fraction` | dacă sub atâta din zonele din față dau o măsurătoare validă (senzor orb, perete lipit de el), se tratează ca obstacol. |
| `max_range` | „nimic în rază” (status 255) se socotește la această distanță. |
| `stop_pause` | cât stă pe loc după oprire, înainte să înceapă să se întoarcă. |
| `min_turn_time`, `min_turn_degrees` | rotirea în loc durează cel puțin atât și parcurge cel puțin atâtea grade **măsurate**. |
| `max_turn_time` | dacă rotirea în loc nu se eliberează în atâta timp, renunță și face întoarcerea mare. |
| `default_turn` | `left` sau `right`: direcția când ambele părți sunt la fel de libere. |

### `[corner]`
| Cheie | Ce face |
|---|---|
| `turns_before_big_turn` | atâtea opriri complete (frână + rotire pe loc) la rând, fără progres între ele => întoarcere mare. Un viraj din mers reușit nu se numără. |
| `progress_time` | cât trebuie să meargă înainte cu drumul liber (peste `slow_distance`), continuu, ca numărătoarea să revină la zero. |
| `big_turn_degrees` | unghiul întoarcerii mari, **măsurat din `odom.yaw`** (nu cronometrat): rotește până atinge unghiul, oricât de mică e rata reală. Nu mai e nevoie de corecția la 280. |
| `big_turn_max_time` | siguranță: dacă unghiul nu e atins în atâta timp (odometrie blocată sau semn greșit), renunță și numără ca expirare. |
| `fallback_turn_efficiency` | doar dacă robotul nu raportează `odom.yaw`: cronometrează presupunând că se rotește real atâta din rata cerută (0,5 măsurat de tine ≈ 0,46–0,7). |

### `[heading]`
Cât merge drept, un regulator PI mic corectează `vyaw` ca `odom.yaw` să rămână la direcția țintă.
Direcția țintă se reia din direcția curentă de fiecare dată când drumul se eliberează după o ocolire.

| Cheie | Ce face |
|---|---|
| `enabled` | pornește/oprește menținerea direcției. |
| `kp`, `ki` | câștigurile (rad/s corecție per rad eroare; per rad·s). `ki` compensează deriva constantă. |
| `max_correction` | corecția maximă adăugată (rad/s). |
| `yaw_sign` | `1` dacă `odom.yaw` crește spre stânga (ca `vyaw`, cum scrie protocolul); `-1` dacă l-ai găsit invers. Dacă pare greșit, întoarcerea mare expiră la `big_turn_max_time`. |

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

Stări: `hold → forward → brake → turn → forward`, cu `bigturn` pentru colțuri. Virajul din mers și
menținerea direcției sunt în `forward`. Un „gardian” din `Brain.tick` stă deasupra tuturor stărilor:
căzut / nesănătos / fără adâncime => `hold`.

**Stare nouă** (dans, odihnă): o clasă cu `name` și `step(brain, inputs) -> (Command, next | None)`,
înregistrată cu `brain.register(...)`. Dacă nu are nevoie de adâncime, `needs_depth = False`.
Detalii în docstring-ul din `brain.py`.

## Limite cunoscute (v0)

* Dacă robotul a căzut, creierul **doar așteaptă**; nu încearcă să-l ridice.
* Regulatorul de direcție și virajul din mers au valori inițiale **neacordate pe mers real**
  (`kp`, `ki`, `steer_yaw_max`, `slow_distance`); testele le verifică doar pe un model simplu.
* Dacă politica ignoră comenzi mici de rotire cât merge, corecția de direcție poate fi înghițită.
* Nu vede obstacole sub nivelul rândurilor folosite (praguri, obiecte joase) și nici sticlă.
* Orientarea senzorului (coloana 0 = stânga) e luată din simulator și din `tofd`; pe robotul real
  merită verificată.
