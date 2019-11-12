import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'toc',
  templateUrl: './toc.component.html',
  styleUrls: ['./toc.component.css']
})
export class TocComponent implements OnInit {
  @Input()
  items: string[] = [];

  constructor() { }

  ngOnInit() {
  }

}
