use crate::engine::Engine;
use crate::model::{Config, Farm, Inventory, Market, Occupant, Private, Tile};
use num_bigint::BigInt;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyString},
};
use rayon::prelude::*;
use serde_json::Value;
fn err(s: impl ToString) -> PyErr {
    PyValueError::new_err(s.to_string())
}
fn value(v: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    match v {
        None => Ok(Value::Null),
        Some(v) => pythonize::depythonize(v).map_err(err),
    }
}
// Fixed schema keys and LOCKED cells are immutable and can share interned
// strings. Every dictionary/list is still newly allocated for each observer.
macro_rules! field {
    ($py:expr, $dict:expr, $key:literal, $value:expr) => {
        $dict.set_item(pyo3::intern!($py, $key), $value)?
    };
}
// Adapt fallible element construction to PyList::new's preallocated builder.
// This streams directly into the final Python list without an intermediate Vec.
struct ListItem<'py>(PyResult<Bound<'py, PyAny>>);
impl<'py> IntoPyObject<'py> for ListItem<'py> {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;
    fn into_pyobject(self, _py: Python<'py>) -> PyResult<Self::Output> {
        self.0
    }
}
fn list_output<'py, T: pyo3::PyTypeInfo>(
    py: Python<'py>,
    items: impl ExactSizeIterator<Item = PyResult<Bound<'py, T>>>,
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, items.map(|item| ListItem(item.map(Bound::into_any))))
}
fn item_key<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyString> {
    macro_rules! known {
        ($($key:literal),*) => { match name {
            $($key => pyo3::intern!(py, $key).clone(),)*
            _ => PyString::new(py, name),
        } };
    }
    known!(
        "WHEAT",
        "CARROT",
        "TOMATO",
        "STRAWBERRY",
        "MELON",
        "EGG",
        "MILK",
        "WOOL",
        "FERTILIZER",
        "GOOSE",
        "COW",
        "SHEEP"
    )
}
fn inventory_output<'py>(py: Python<'py>, inventory: &Inventory) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    // Iteration order is part of inventory semantics (overflow drops items in it).
    for (name, amount) in inventory {
        dict.set_item(item_key(py, name), amount)?;
    }
    Ok(dict)
}
fn private_output<'py>(py: Python<'py>, private: &Private) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    field!(py, dict, "shed", inventory_output(py, &private.shed)?);
    field!(py, dict, "seeds", inventory_output(py, &private.seeds)?);
    field!(
        py,
        dict,
        "inventories",
        list_output(
            py,
            private.inventories.iter().map(|v| inventory_output(py, v))
        )?
    );
    Ok(dict)
}
fn market_output<'py>(py: Python<'py>, market: &Market) -> PyResult<Bound<'py, PyDict>> {
    fn numbers<'py>(
        py: Python<'py>,
        values: &indexmap::IndexMap<String, serde_json::Number>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in values {
            let key = item_key(py, key);
            // Test integer representations before f64: byte parity distinguishes
            // 1 from 1.0, and u64 values must not round through floating point.
            if let Some(value) = value.as_i64() {
                dict.set_item(key, value)?;
            } else if let Some(value) = value.as_u64() {
                dict.set_item(key, value)?;
            } else {
                dict.set_item(key, value.as_f64().unwrap())?;
            }
        }
        Ok(dict)
    }
    let dict = PyDict::new(py);
    field!(py, dict, "inventory", numbers(py, &market.inventory)?);
    field!(py, dict, "prices", numbers(py, &market.prices)?);
    if let Some(params) = &market.params {
        field!(py, dict, "params", pythonize::pythonize(py, params)?);
    }
    Ok(dict)
}
fn occupant_output<'py>(py: Python<'py>, occupant: &Occupant) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match occupant {
        Occupant::Plant(plant) => {
            field!(py, dict, "kind", pyo3::intern!(py, "PLANT"));
            field!(py, dict, "crop", item_key(py, &plant.crop));
            field!(py, dict, "planted_day", plant.planted_day);
            field!(py, dict, "watered_today", plant.watered_today);
            field!(
                py,
                dict,
                "consecutive_unwatered",
                plant.consecutive_unwatered
            );
            field!(py, dict, "yield_units", plant.yield_units);
            field!(py, dict, "max_lifespan_step", plant.max_lifespan_step);
            field!(py, dict, "fertilized_until_day", plant.fertilized_until_day);
        }
        Occupant::Weed => {
            field!(py, dict, "kind", pyo3::intern!(py, "WEED"));
        }
        Occupant::Coop { animal } | Occupant::Pasture { animal } => {
            let kind = if matches!(occupant, Occupant::Coop { .. }) {
                pyo3::intern!(py, "COOP")
            } else {
                pyo3::intern!(py, "PASTURE")
            };
            field!(py, dict, "kind", kind);
            if let Some(animal) = animal {
                field!(py, dict, "animal", item_key(py, &animal.animal));
                field!(py, dict, "placed_day", animal.placed_day);
                field!(py, dict, "yield_units", animal.yield_units);
                field!(py, dict, "consecutive_unfed", animal.consecutive_unfed);
                field!(py, dict, "fed_today", animal.fed_today);
                field!(py, dict, "cared_today", animal.cared_today);
                field!(
                    py,
                    dict,
                    "fertilizer_available",
                    animal.fertilizer_available
                );
                field!(py, dict, "pending_care_bonus", animal.pending_care_bonus);
            }
        }
    }
    Ok(dict)
}
fn farm_output<'py>(py: Python<'py>, farm: &Farm) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    field!(py, dict, "money", farm.money);
    let rows = list_output(
        py,
        farm.tiles.iter().map(|row| {
            list_output(
                py,
                row.iter().map(|tile| -> PyResult<Bound<'py, PyAny>> {
                    Ok(match tile {
                        Tile::Empty => py.None().into_bound(py),
                        Tile::Locked(s) if s == "LOCKED" => {
                            pyo3::intern!(py, "LOCKED").clone().into_any()
                        }
                        Tile::Locked(s) => PyString::new(py, s).into_any(),
                        Tile::Occupied(occupant) => occupant_output(py, occupant)?.into_any(),
                    })
                }),
            )
        }),
    )?;
    field!(py, dict, "tiles", rows);
    field!(py, dict, "farmer", PyList::new(py, farm.farmer)?);
    field!(
        py,
        dict,
        "hands",
        list_output(
            py,
            farm.hands.iter().map(|position| PyList::new(py, *position))
        )?
    );
    field!(
        py,
        dict,
        "unlocked_quadrants",
        PyList::new(py, &farm.unlocked_quadrants)?
    );
    field!(py, dict, "hires_today", farm.hires_today);
    Ok(dict)
}
fn observations_output<'py>(py: Python<'py>, engine: &Engine) -> PyResult<Bound<'py, PyList>> {
    list_output(
        py,
        (0..2).map(|player| {
            let dict = PyDict::new(py);
            field!(py, dict, "player", player);
            field!(py, dict, "step", engine.step);
            field!(py, dict, "day", engine.step / engine.config.turns);
            field!(py, dict, "hour", engine.step % engine.config.turns);
            field!(
                py,
                dict,
                "farms",
                list_output(py, engine.farms.iter().map(|farm| farm_output(py, farm)))?
            );
            field!(
                py,
                dict,
                "private",
                private_output(py, &engine.privates[player])?
            );
            field!(py, dict, "market", market_output(py, engine.market())?);
            let town = PyDict::new(py);
            field!(
                py,
                town,
                "unlocked_shops",
                PyList::new(py, &engine.town.unlocked_shops)?
            );
            field!(py, dict, "town", town);
            field!(py, dict, "remainingOverageTime", 60);
            Ok(dict)
        }),
    )
}
fn result_output<'py>(py: Python<'py>, engine: &Engine) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    field!(py, dict, "observations", observations_output(py, engine)?);
    // Preserve integer zero before terminal and serde_json's conversion of
    // terminal money (including its handling of non-finite values).
    let rewards = if engine.done {
        list_output(
            py,
            engine.farms.iter().map(|farm| {
                pythonize::pythonize(py, &Value::from(farm.money)).map_err(PyErr::from)
            }),
        )?
    } else {
        PyList::new(py, [0, 0])?
    };
    field!(py, dict, "rewards", rewards);
    let status = if engine.done {
        pyo3::intern!(py, "DONE")
    } else {
        pyo3::intern!(py, "ACTIVE")
    };
    field!(py, dict, "statuses", PyList::new(py, [status, status])?);
    field!(py, dict, "done", engine.done);
    Ok(dict)
}
fn batch_results(py: Python<'_>, envs: &[Engine]) -> PyResult<Py<PyAny>> {
    Ok(
        list_output(py, envs.iter().map(|env| result_output(py, env)))?
            .into_any()
            .unbind(),
    )
}

fn pair(v: Value) -> PyResult<[Value; 2]> {
    let Value::Array(a) = v else {
        return Err(err("actions must be a pair"));
    };
    a.try_into()
        .map_err(|_| err("exactly two player actions required"))
}
#[pyclass(name = "Env")]
struct PyEnv {
    inner: Engine,
}
#[pymethods]
impl PyEnv {
    #[new]
    #[pyo3(signature=(config=None,seed=BigInt::from(0)))]
    fn new(config: Option<&Bound<'_, PyAny>>, seed: BigInt) -> PyResult<Self> {
        let v = value(config)?;
        let c = Config::parse(&v).map_err(err)?;
        Ok(Self {
            inner: Engine::new(c, seed),
        })
    }
    fn reset(&mut self, py: Python<'_>, seed: BigInt) -> PyResult<Py<PyAny>> {
        self.inner = Engine::new(self.inner.config.clone(), seed);
        Ok(observations_output(py, &self.inner)?.into_any().unbind())
    }
    /// Return a detached copy of the current state without advancing the game.
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(result_output(py, &self.inner)?.into_any().unbind())
    }
    /// An independent copy of the engine in its current state -- for branching a game
    /// (policy improvement by rollouts). The copy and
    /// the original live separately from then on: a step of one does not touch the other. The engine
    /// is deterministic given the seed and actions (its own day RNG inside `Engine`),
    /// so identical actions in the copy and the original give bit-for-bit the same
    /// game -- this is checked by a branching probe script.
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
    fn __copy__(&self) -> Self {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let a = pair(value(Some(actions))?)?;
        py.detach(|| self.inner.advance(&a));
        Ok(result_output(py, &self.inner)?.into_any().unbind())
    }

    /// Tile features of both farms from the view of `player`: the raw bytes of two
    /// float32 arrays `[board*board, TILE_FEAT_DIM]` back to back -- own farm first,
    /// then the opponent's (`tiles_me`, `tiles_op` in terms of `features.encode`).
    ///
    /// Returned as bytes, not via `numpy`: an extra dependency of the Rust crate
    /// for a single `memcpy` does not pay off, and in Python this is
    /// `np.frombuffer(b, dtype=np.float32).reshape(2, T, F)` -- with no copy beyond
    /// the one `bytes` itself makes.
    ///
    /// ⚠ Reads the engine state DIRECTLY, bypassing the observation: that is the whole
    /// point (Rust -> Rust, no bridge). So the value must match the
    /// Python `features.encode` to the bit -- checked by an
    /// encode parity script.
    fn encode_tiles<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyBytes>> {
        if player > 1 {
            return Err(err("player must be 0 or 1"));
        }
        let board = self.inner.config.board;
        let turns = self.inner.config.turns.max(1);
        let day = self.inner.step / turns;
        let n = board * board * crate::encode::TILE_FEAT_DIM;
        let mut buf = vec![0f32; 2 * n];
        let step = self.inner.step;
        let (me, op) = buf.split_at_mut(n);
        crate::encode::farm_tiles(me, &self.inner.farms[player], day, board, step);
        crate::encode::farm_tiles(op, &self.inner.farms[1 - player], day, board, step);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4)
        };
        Ok(PyBytes::new(py, bytes))
    }

    /// The whole `features.encode` in one call. -> raw float32 bytes in
    /// the layout of `encode::layout()`; Python parses them with `np.frombuffer`.
    ///
    /// `extra` -- 36 numbers of the Python `FlowTracker` (pi_, oppsold_, mysold_,
    /// crowd_ over nine goods, in `PRODUCTS` order). This is the only thing that
    /// comes from outside: the rest is read from the engine state directly.
    #[pyo3(signature=(player, extra))]
    fn encode_all<'py>(
        &self,
        py: Python<'py>,
        player: usize,
        extra: Vec<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if player > 1 {
            return Err(err("player must be 0 or 1"));
        }
        let n = crate::model::PRODUCTS.len();
        if extra.len() != 4 * n {
            return Err(err(format!("extra must be {} floats", 4 * n)));
        }
        let ex = crate::encode::Extra {
            pi: &extra[0..n],
            oppsold: &extra[n..2 * n],
            mysold: &extra[2 * n..3 * n],
            crowd: &extra[3 * n..4 * n],
        };
        let mut buf = vec![0f32; crate::encode::flat_len()];
        let shed = crate::encode::shed_tiles_of(self.inner.config.board);
        crate::encode::encode_all(
            &mut buf,
            &self.inner.config,
            &self.inner.farms,
            &self.inner.privates,
            self.inner.market(),
            self.inner.pricing(),
            &self.inner.town,
            self.inner.step,
            player,
            &ex,
            &shed,
            crate::model::shops_def(),
        );
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4) };
        Ok(PyBytes::new(py, bytes))
    }

    /// The engine's current turn number. Needed by the native encoder: the slot
    /// `features.NATIVE` is per-process, and the closure must make sure it
    /// is being asked about THIS state, not someone else's observation.
    fn current_step(&self) -> i64 {
        self.inner.step
    }

    /// Layout of the flat `encode_all` output: [(name, offset, length)].
    #[staticmethod]
    fn encode_layout() -> Vec<(String, usize, usize)> {
        crate::encode::layout()
            .into_iter()
            .map(|(n, o, k)| (n.to_string(), o, k))
            .collect()
    }
}
#[pyclass(name = "BatchEnv")]
struct PyBatch {
    envs: Vec<Engine>,
    pool: rayon::ThreadPool,
}
#[pymethods]
impl PyBatch {
    #[new]
    #[pyo3(signature=(num_envs,config=None,num_threads=None))]
    fn new(
        num_envs: usize,
        config: Option<&Bound<'_, PyAny>>,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(err("num_envs must be positive"));
        }
        if num_threads == Some(0) {
            return Err(err("num_threads must be positive"));
        }
        let c = Config::parse(&value(config)?).map_err(err)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(1)
            }))
            .build()
            .map_err(err)?;
        Ok(Self {
            envs: (0..num_envs)
                .map(|i| Engine::new(c.clone(), BigInt::from(i)))
                .collect(),
            pool,
        })
    }
    fn reset(&mut self, py: Python<'_>, seeds: Vec<BigInt>) -> PyResult<Py<PyAny>> {
        if seeds.len() != self.envs.len() {
            return Err(err("one seed per environment required"));
        }
        py.detach(|| {
            if self.envs.len() == 1 || self.pool.current_num_threads() == 1 {
                for (e, s) in self.envs.iter_mut().zip(seeds) {
                    *e = Engine::new(e.config.clone(), s);
                }
            } else {
                self.pool.install(|| {
                    self.envs
                        .par_iter_mut()
                        .zip(seeds)
                        .for_each(|(e, s)| *e = Engine::new(e.config.clone(), s))
                });
            }
        });
        Ok(
            list_output(py, self.envs.iter().map(|env| observations_output(py, env)))?
                .into_any()
                .unbind(),
        )
    }
    /// Return current results in environment order without advancing any slot.
    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        batch_results(py, &self.envs)
    }
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let v = value(Some(actions))?;
        let Value::Array(batch) = v else {
            return Err(err("actions must be a batch"));
        };
        if batch.len() != self.envs.len() {
            return Err(err("one action pair per environment required"));
        }
        // Validate the entire batch before advancing any environment. Moving the
        // owned values avoids cloning every nested order twice.
        let a: Vec<[Value; 2]> = batch.into_iter().map(pair).collect::<PyResult<_>>()?;
        py.detach(|| {
            if self.envs.len() == 1 || self.pool.current_num_threads() == 1 {
                for (e, a) in self.envs.iter_mut().zip(&a) {
                    e.advance(a);
                }
            } else {
                self.pool.install(|| {
                    self.envs
                        .par_iter_mut()
                        .zip(&a)
                        .for_each(|(e, a)| e.advance(a))
                });
            }
        });
        batch_results(py, &self.envs)
    }
    fn reset_at(
        &mut self,
        py: Python<'_>,
        indices: Vec<usize>,
        seeds: Vec<BigInt>,
    ) -> PyResult<Py<PyAny>> {
        if indices.len() != seeds.len() {
            return Err(err("indices and seeds must have equal length"));
        }
        let mut seen = std::collections::HashSet::new();
        for &i in &indices {
            if i >= self.envs.len() || !seen.insert(i) {
                return Err(err("indices must be unique and in range"));
            }
        }
        py.detach(|| {
            for (&i, s) in indices.iter().zip(seeds) {
                self.envs[i] = Engine::new(self.envs[i].config.clone(), s);
            }
        });
        Ok(list_output(
            py,
            indices
                .iter()
                .map(|&i| observations_output(py, &self.envs[i])),
        )?
        .into_any()
        .unbind())
    }
}
#[pymodule]
fn kaggriculture_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnv>()?;
    m.add_class::<PyBatch>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "REFERENCE_COMMIT",
        "28b6d8af3ce73926b3d0fda1410c1ddd8384ab8c",
    )?;
    Ok(())
}
