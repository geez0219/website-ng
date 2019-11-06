import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'toc',
  templateUrl: './toc.component.html',
  styleUrls: ['./toc.component.css']
})
export class TocComponent implements OnInit {
  items: string[];

  constructor() { }

  ngOnInit() {
    this.items = ["asd", "asdf"]
  }

}
