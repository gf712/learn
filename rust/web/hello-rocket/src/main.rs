#![feature(proc_macro_hygiene, decl_macro)]

#[macro_use] extern crate rocket;
use rocket::http::RawStr;

mod other {
	#[get("/world")]
	pub fn world() -> &'static str {
		"Hello, World!"
	}
}

#[get("/hello/<name>")]
fn hello(name: &RawStr) -> String {
    format!("Hello, {}!", name)
}

fn main() {
    rocket::ignite().mount("/", routes![hello, other::world]).launch();
}